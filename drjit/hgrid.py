from __future__ import annotations
from dataclasses import dataclass
import dataclasses
import sys
import drjit

if sys.version_info < (3, 11):
    from typing_extensions import Tuple, Type, Any, overload, Iterable
else:
    from typing import Tuple, Type, Any, overload, Iterable


def cosine_ramp(x):
    """ "Smoothed" ramp to help features blend-in without instabilities"""
    return 0.5 * (1.0 - drjit.cos(drjit.pi * x))

def div_round_up(num: int, divisor: int) -> int:
    return (num + divisor - 1) // divisor


def next_multiple(num: int, multiple: int) -> int:
    return multiple * div_round_up(num, multiple)

@dataclass
class HashEncodingConfig:
    """
    This class holds information about a hash based encoding, such as hash grids
    or permutohedral encodings.

    Args:
        dimension: The dimensionality of the hash encoding. This corresponds to
            the number of input features the encoding can take.
        n_levels: Hash encodings generally make use of multiple levels of the same
            encoding with different scales. This parameter specifies the number of
            levels used by this encoding.
        n_features_per_level: More than one feature can be stored in a vertex per
            level. This value specifies how many, and the number of output features
            of the hash encoding layer is given by ``n_levels * n_features_per_level``.
            This value should always be a multiple of two, in order to ensure efficient
            gradient backpropagation.
        hashmap_size: Specifies the maximal number of parameters per level of the
            hash encoding. HashGrids will use a dense grid lookup for layers with
            a low enough scale, and use less than ``hashmap_size`` number of parameters
            per level.
        base_resolution: The scale factor of the 0th layer in the hash encoding.
        per_level_scale: To calculate the scale of a layer, the scale of the previous
            layer is multiplied by this value.
        align_corners: If this value is ``True``, the simplex vertices are aligned
            with the domain of the encoding [0, 1].
        smooth_weight_gradients: whether to smooth the gradients of the weights
            by using a straight-through estimator.
        smooth_weight_lambda: the value of lambda used for the straight-through estimator.
    """
    dimension: int
    n_levels: int
    n_features_per_level: int
    hashmap_size: int
    base_resolution: int
    per_level_scale: float
    align_corners: bool
    torchngp_compat: bool
    smooth_weight_gradients: bool
    smooth_weight_lambda: float

    def __init__(
        self,
        dimension: int = -1,
        n_levels: int = -1,
        n_features_per_level: int = -1,
        hashmap_size: int = 2**19,
        base_resolution: int = 16,
        per_level_scale: float = 2,
        align_corners: bool = False,
        torchngp_compat: bool = False,
        smooth_weight_gradients: bool = False,
        smooth_weight_lambda: float = 1.0,
    ) -> None:
        self.dimension = dimension
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.hashmap_size = hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.log2_per_level_scale = drjit.log2(per_level_scale)
        self.align_corners = align_corners
        self.torchngp_compat = torchngp_compat
        self.smooth_weight_gradients = smooth_weight_gradients
        self.smooth_weight_lambda = smooth_weight_lambda


class HashEncoding:
    """
    This class serves as the interface for Hash based encodings.
    """
    DRJIT_STRUCT = {
        "data": drjit.ArrayBase,
        "dtype": type,
        "_config": HashEncodingConfig,
        "_level_offsets": list,
        "_cell_sizes": list,
    }

    def alloc(
        self, dtype: Type[drjit.ArrayBase], /
    ) -> HashEncoding:

        result = type(self)()
        result._config = dataclasses.replace(self._config)
        result._alloc(dtype)

        return result

    def n_params(self) -> int:
        return drjit.width(self.data)

    def set_params(self, values: drjit.ArrayBase, copy=False) -> None:
        """
        Should pass `values` as a DrJit array (float32 only for now).
        """
        assert drjit.width(values) == drjit.width(self.data)
        assert type(values) is type(self.data)
        if copy:
            self.data = type(self.data)(values)
        else:
            assert isinstance(values, type(self.data))
            self.data = values

    @property
    def dimension(self) -> int:
        """
        The dimensionality of this hash encoding.
        """
        return self._config.dimension

    @property
    def hashmap_size(self) -> int:
        """
        The hashmap size provided when constructing this encoding.
        """
        return self._config.hashmap_size

    @property
    def n_levels(self) -> int:
        """
        The number of levels in this hash encoding.
        The actual number of output features of this encoding is determined by
        ``n_level * n_features_per_level``.
        """
        return self._config.n_levels

    @property
    def base_resolution(self) -> int:
        """
        The resolution of the 0th level.
        """
        return self._config.base_resolution

    @property
    def per_level_scale(self) -> float:
        """
        The per level scale factor, with which the scale of each level grows.
        """
        return self._config.per_level_scale

    @property
    def n_features_per_level(self) -> int:
        """
        The number of features per level.
        The actual number of output features of this encoding is determined by
        ``n_level * n_features_per_level``.
        """
        return self._config.n_features_per_level

    @property
    def align_corners(self) -> bool:
        """
        If the corners of the hashgrid should be aligned to the edges of its domain.
        """
        return self._config.align_corners

    @property
    def torchngp_compat(self) -> int:
        return self._config.torchngp_compat

    @property
    def smooth_weight_gradients(self) -> bool:
        return self._config.smooth_weight_gradients

    @property
    def smooth_weight_lambda(self) -> float:
        return self._config.smooth_weight_lambda

    @property
    def out_features(self) -> int:
        return self._config.n_features_per_level *  self.n_levels

    def _init_types(self, p):
        """
        Gets the type aliases used for this hash encoding computation, given a
        position value ``p``.
        """
        dtype = self.dtype
        mod = sys.modules[dtype.__module__]
        self.PCG32 = mod.PCG32
        self.StorageFloat = dtype
        self.UInt32 = drjit.uint32_array_t(dtype)
        self.Int32 = drjit.int32_array_t(dtype)
        self.ArrayXu = mod.ArrayXu
        self.ArrayXi = mod.ArrayXi
        self.StorageFloatXf = mod.ArrayXf16 if drjit.is_half_v(dtype) else mod.ArrayXf

        p = list(p)

        if isinstance(p, list):
            self.PositionFloat = drjit.leaf_t(p[0])
        else:
            self.PositionFloat = drjit.leaf_t(p)

        self.PositionFloatXf = mod.ArrayXf16 if drjit.is_half_v(self.PositionFloat) else mod.ArrayXf

    def _acc_features(self, level_i, weight, index, values: list, active):
        """
        Accumulates the ``self.num_features`` features at the given index.
        This function tries to use packet gather operations if possible, to improve
        the performance of the backward pass, as atomic packet scatters
        perform better than their non packeted counterparts.
        """

        if self.smooth_weight_gradients:
            weight_smooth = cosine_ramp(weight)
            weight = weight + self.smooth_weight_lambda * (weight_smooth - drjit.detach(weight_smooth))

        # TODO: warning for 3 features
        v = drjit.gather(
            self.StorageFloatXf,
            self.data,
            index // self.n_features_per_level,
            active,
            shape=(self.n_features_per_level, drjit.width(index)),
        )

        for k in range(0, self.n_features_per_level):
            values[level_i * self.n_features_per_level + k] = drjit.fma(
                v[k], weight, values[level_i * self.n_features_per_level + k]
            )

    def indexing_function(self, key, level_i):
        raise NotImplementedError()


class HashGridEncoding(HashEncoding):
    """
    This encoding implements a Multiresolution Hash Grid. For every resolution level,
    this encoding looks up the :math:`2^D` vertices of the cell in which the input point is
    located, performs multilinear interpolation, and concatenates the features accross
    all resolution levels.

    Args:
        dimension: The dimensionality of the hash encoding. This corresponds to
            the number of input features the encoding can take.
        n_levels: Hash encodings generally make use of multiple levels of the same
            encoding with different scales. This parameter specifies the number of
            levels used by this encoding.
        n_features_per_level: More than one feature can be stored in a vertex per
            level. This value specifies how many, and the number of output features
            of the hash encoding layer is given by ``n_levels * n_features_per_level``.
            This value should always be a multiple of two, in order to ensure efficient
            gradient backpropagation.
        hashmap_size: Specifies the maximal number of parameters per level of the
            hash encoding. HashGrids will use a dense grid lookup for layers with
            a low enough scale, and use less than ``hashmap_size`` number of parameters
            per level.
        base_resolution: The scale factor of the 0th layer in the hash encoding.
        per_level_scale: To calculate the scale of a layer, the scale of the previous
            layer is multiplied by this value.
        align_corners: If this value is ``True``, the simplex vertices are aligned
            with the domain of the encoding [0, 1].
        smooth_weight_gradients: whether to smooth the gradients of the weights
            by using a straight-through estimator.
        smooth_weight_lambda: the value of lambda used for the straight-through estimator.
    """
    DRJIT_STRUCT = {
        "data": drjit.ArrayBase,
        "dtype": type,
        "_config": HashEncodingConfig,
        "_level_offsets": list,
        "_cell_sizes": list,
        "_grid_offsets": list,
    }

    @overload
    def __init__(
        self,
        dimension: int = -1,
        n_levels: int = -1,
        n_features_per_level: int = -1,
        *,
        hashmap_size: int = 2**19,
        base_resolution: int = 16,
        per_level_scale: float = 2,
        align_corners: bool = False,
        torchngp_compat: bool = False,
        smooth_weight_gradients: bool = False,
        smooth_weight_lambda: float = 1.0,
    ) -> None: ...

    @overload
    def __init__(self) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        """
        This encoding is based on the Multiresolution Hash Grid encoding introduced
        in :cite:`mueller2022instant`.

        Args:
            dimension: The dimensionality of the hash encoding. This corresponds to
                the number of input features the encoding can take.
            n_levels: Hash encodings generally make use of multiple levels of the same
                encoding with different scales. This parameter specifies the number of
                levels used by this encoding.
            n_features_per_level: More than one feature can be stored in a vertex per
                level. This value specifies how many, and the number of output features
                of the hash encoding layer is given by ``n_levels * n_features_per_level``.
                This value should always be a multiple of two, in order to ensure efficient
                gradient backpropagation.
            hashmap_size: Specifies the maximal number of parameters per level of the
                hash encoding. HashGrids will use a dense grid lookup for layers with
                a low enough scale, and use less than ``hashmap_size`` number of parameters
                per level.
            base_resolution: The scale factor of the 0th layer in the hash encoding.
            per_level_scale: To calculate the scale of a layer, the scale of the previous
                layer is multiplied by this value.
            align_corners: If this value is ``True``, the simplex vertices are aligned
                with the domain of the encoding [0, 1].
            smooth_weight_gradients: whether to smooth the gradients of the weights
                by using a straight-through estimator.
            smooth_weight_lambda: the value of lambda used for the straight-through estimator.
        """
        if len(args) == 0:
            self._config = HashEncodingConfig()
        else:
            self._config = HashEncodingConfig(*args, **kwargs)

    def __call__(
        self, p: Iterable[drjit.ArrayBase], active=True
    ) -> Iterable[drjit.ArrayBase]:
        self._init_types(p)

        p = self.PositionFloatXf(p)

        assert drjit.shape(p)[0] == self.dimension, (
            f"This hash grid expected an input of feature dimension {self.dimension}"
            f" but got {drjit.shape(p)[0]}."
        )

        values = [self.StorageFloat(0.0)] * self.n_features_per_level * self.n_levels

        for level_i in range(self.n_levels):
            scale = self._grid_scale(level_i)

            p_offset: float = 0.0 if self.align_corners else 0.5
            pos = drjit.fma(p, scale, p_offset)

            pos0 = self.ArrayXu(drjit.floor(pos))

            w1 = pos - pos0
            w0 = 1.0 - w1

            for offset in self._grid_offsets:
                pos_grid = pos0 + self.ArrayXu(offset)
                weight = drjit.select(self.ArrayXu(offset) == 0, w0, w1)
                weight = drjit.prod(weight, axis=0)

                index = self.indexing_function(pos_grid, level_i)
                self._acc_features(level_i, weight, index, values, active)

        values = [v & active for v in values]

        return self.StorageFloatXf(*values)

    def indexing_function(self, key, level_i):
        """
        This function is used to index the underlying data array of the
        hash grid given a grid position.
        """

        scale = self._grid_scale(level_i)
        res = self._grid_resolution(scale)
        level_offset = self._level_offsets[level_i]
        this_level_size = self._level_offsets[level_i + 1] - level_offset

        indexing_primes = [1, self.UInt32(2654435761), self.UInt32(805459861)]

        stride = 1
        index = self.UInt32(0)
        for d in range(self.dimension):
            index += key[d] * stride
            stride *= res

            if stride > this_level_size:
                break

        if this_level_size < stride:
            index = self.UInt32(0)
            for d in range(0, self.dimension):
                index ^= key[d] * indexing_primes[d]

        sub_grid_index = level_offset + (index % self.UInt32(this_level_size))
        return self.n_features_per_level * sub_grid_index

    def _alloc(self, dtype: Type[drjit.ArrayBase]):
        self.dtype = drjit.leaf_t(dtype)

        assert (self.hashmap_size % 8) == 0, (
            f"Invalid hashmap size {self.hashmap_size}, must be a multiple of 8."
        )

        self._level_offsets = [None] * (self.n_levels + 1)
        self._cell_sizes = [0.0] * self.n_levels
        max_params = drjit.scalar.UInt32(0xFFFFFFFF) // 2
        offset = 0

        for level_i in range(self.n_levels):
            res = self._grid_resolution(self._grid_scale(level_i))
            stride = drjit.power(float(res), self.dimension)

            params_in_level = (
                int(max_params) if (stride > max_params) else (res**self.dimension)
            )
            params_in_level = next_multiple(params_in_level, 8)
            params_in_level = min(params_in_level, self.hashmap_size)

            self._level_offsets[level_i] = offset
            self._cell_sizes[level_i] = 1.0 / (
                self.base_resolution * self.per_level_scale**level_i
            )
            offset += params_in_level

        self._level_offsets[-1] = offset

        params_size = self._level_offsets[-1] * self.n_features_per_level
        self.data = drjit.zeros(dtype, params_size)

        # Stores the pattern of offsets used to index the 2** n corners of a voxel
        self._grid_offsets = [
            [(i >> j) & 1 for j in range(self.dimension)]
            for i in range(2**self.dimension)
        ]

    def _grid_scale(self, level) -> float:
        """
        The -1 means that `base_resolution` refers to the number of grid _vertices_ rather
        than the number of cells. This is slightly different from the notation in the paper,
        but results in nice, power-of-2-scaled parameter grids that fit better into cache lines.
        """
        return (
            drjit.exp2(level * self._config.log2_per_level_scale) * self.base_resolution
            - 1.0
        )

    def _grid_resolution(self, scale) -> int:
        return (
            int(drjit.ceil(scale))
            + (0 if self.align_corners else 1)
            + (1 if self.torchngp_compat else 0)
        )

    def __repr__(self) -> str:
        return (
            f"HashGrid(\n"
            f"    dtype={self.dtype},\n"
            f"    dimension={self.dimension},\n"
            f"    hashmap_size={self.hashmap_size},\n"
            f"    n_levels={self.n_levels},\n"
            f"    n_features_per_level={self.n_features_per_level},\n"
            f"    base_resolution={self.base_resolution},\n"
            f"    per_level_scale={self.per_level_scale},\n"
            f"    align_corners={self.align_corners},\n"
            f"    torchngp_compat={self.torchngp_compat},\n"
            f"    smooth_weight_gradients={self.smooth_weight_gradients},\n"
            f"    smooth_weight_lambda={self.smooth_weight_lambda}\n"
            ")"
        )

    # --------------------

    def default_padding_value(self) -> float:
        return 0.0

    def n_dims_encoded(self, n_dims_in: int) -> int:
        return self.n_features_per_level * self.n_levels

    # --------------------

    def level_offset(self, level_i: int) -> int:
        """
        Helpful to build e.g. debug visualizations by splitting params per level
        Must be called with an index up to and including `n_levels`.

        Warning: the level offset returned does *not* account for the feature count.
        Each level contains `num_features()` times the difference between the next
        offset entries.
        """
        return self._level_offsets[level_i]


class SimplifiedPermutohedralEncoding(HashEncoding):
    """
    This encoding is based on the encoding presented in :cite:`rosu2023permutosdf`.
    Whereas hash grid encodings use a grid lattice, this uses a permutohedral lattice,
    where the simplexes are consist of triangles, tetrahedrons etc. The main advantage
    is that the number of vertices per simplex and therefore the number of memory
    lookups per sample per layer grow linear with the number of dimensions.
    In this implementation we make a few simplification, and skip the elevation to a
    hyperplane in d + 1 dimensional space. It instead performs the sorting and interpolation
    steps in d-dimensional space.

    Args:
        dimension: The dimensionality of the hash encoding. This corresponds to
            the number of input features the encoding can take.
        n_levels: Hash encodings generally make use of multiple levels of the same
            encoding with different scales. This parameter specifies the number of
            levels used by this encoding.
        n_features_per_level: More than one feature can be stored in a vertex per
            level. This value specifies how many, and the number of output features
            of the hash encoding layer is given by ``n_levels * n_features_per_level``.
            This value should always be a multiple of two, in order to ensure efficient
            gradient backpropagation.
        hashmap_size: Specifies the maximal number of parameters per level of the
            hash encoding. HashGrids will use a dense grid lookup for layers with
            a low enough scale, and use less than ``hashmap_size`` number of parameters
            per level.
        base_resolution: The scale factor of the 0th layer in the hash encoding.
        per_level_scale: To calculate the scale of a layer, the scale of the previous
            layer is multiplied by this value.
        align_corners: If this value is ``True``, the simplex vertices are aligned
            with the domain of the encoding [0, 1].
        smooth_weight_gradients: whether to smooth the gradients of the weights
            by using a straight-through estimator.
        smooth_weight_lambda: the value of lambda used for the straight-through estimator.
    """

    DRJIT_STRUCT = {
        "data": drjit.ArrayBase,
        "dtype": type,
        "_config": HashEncodingConfig,
        "_level_offsets": list,
    }

    @overload
    def __init__(
        self,
        dimension: int,
        n_levels: int,
        n_features_per_level: int,
        *,
        hashmap_size: int = 2**19,
        base_resolution: int = 16,
        per_level_scale: float = 2,
        smooth_weight_gradients: bool = False,
        smooth_weight_lambda: float = 1.0,
    ) -> None: ...

    @overload
    def __init__(self) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 0:
            self._config = HashEncodingConfig()
        else:
            self._config = HashEncodingConfig(*args, **kwargs)

    def __call__(
        self, p: Iterable[drjit.ArrayBase], active=True
    ) -> Any:
        self._init_types(p)

        p = self.PositionFloatXf(p)

        assert drjit.shape(p)[0] == self.dimension, (
            f"This permutohedral grid expected an input of feature dimension {self.dimension}"
            f" but got {drjit.shape(p)[0]}."
        )

        values = [self.StorageFloat(0.0)] * self.n_features_per_level * self.n_levels

        for level_i in range(self.n_levels):
            scale = self._level_scale(level_i)

            # ---- Apply scaling factor

            p_offset: float = 0.0 if self.align_corners else 0.5
            x = drjit.fma(p, scale, p_offset)

            base = self.ArrayXi(drjit.floor(x))
            fract = x - base

            # ---- Insertion sort fract into xk and indices xki
            xk = self.PositionFloatXf(fract)
            xki = self.ArrayXi([i for i in range(self.dimension)])

            for i in range(self.dimension):
                for j in range(i, self.dimension):
                    xk_i = self.PositionFloat(xk[i])
                    xk_j = self.PositionFloat(xk[j])
                    swap = xk_i > xk_j
                    xk[i] = drjit.select(swap, xk_j, xk_i)
                    xk[j] = drjit.select(swap, xk_i, xk_j)

                    xki_i = self.Int32(xki[i])
                    xki_j = self.Int32(xki[j])
                    xki[i] = drjit.select(swap, xki_j, xki_i)
                    xki[j] = drjit.select(swap, xki_i, xki_j)

            # ---- Calculate Barycentric coordinates and grid offset vectors
            grid_offsets = [None] * (self.dimension + 1)
            weights = drjit.zeros(
                self.PositionFloatXf, shape=(self.dimension + 1, drjit.width(p))
            )

            for rank in range(self.dimension):
                weights[rank] = xk[rank] - (xk[rank - 1] if rank > 0 else 0)
                grid_offset = drjit.select(
                    (fract > xk[rank])
                    | (fract == xk[rank])
                    & (self.ArrayXi([i for i in range(self.dimension)]) >= xki[rank]),
                    1,
                    0,
                )
                grid_offsets[rank] = grid_offset

            weights[self.dimension] = 1 - xk[self.dimension - 1]
            grid_offsets[self.dimension] = drjit.zeros(
                self.ArrayXi, shape=(self.dimension, drjit.width(p))
            )

            # ---- Accumulate features for each offset vector
            for rank in range(self.dimension + 1):
                offset = grid_offsets[rank]

                pos_grid = base + self.ArrayXu(offset)
                weight = weights[rank]

                index = self.indexing_function(pos_grid, level_i)
                self._acc_features(level_i, weight, index, values, active)

        values = [v & active for v in values]

        return self.StorageFloatXf(*values)

    def _alloc(self, dtype: Type[drjit.ArrayBase]):
        self.dtype = drjit.leaf_t(dtype)

        assert (self.hashmap_size % 8) == 0, (
            f"Invalid hashmap size {self.hashmap_size}, must be a multiple of 8."
        )

        self._level_offsets = [None] * (self.n_levels + 1)

        offset = 0
        for i in range(self.n_levels):
            params_in_level = self.hashmap_size
            self._level_offsets[i] = offset
            offset += params_in_level

        self._level_offsets[-1] = offset

        params_size = self._level_offsets[-1] * self.n_features_per_level
        self.data = drjit.zeros(dtype, params_size)

    def _level_scale(self, level) -> float:
        return (
            drjit.exp2(level * self._config.log2_per_level_scale) * self.base_resolution
        )

    def _resolution(self, scale) -> int:
        return int(drjit.ceil(scale))

    def indexing_function(self, key, level_i):
        """
        This function is used to index the underlying data array of the
        hash grid given a grid position.
        """

        def base_convert_hash(key: self.ArrayXu) -> self.UInt32:
            k = self.UInt32(0)
            for dim in range(self.dimension):
                k += key[dim]
                k *= 2531011
            return k

        level_offset = self._level_offsets[level_i]
        this_level_size = self._level_offsets[level_i + 1] - level_offset

        index = base_convert_hash(key)
        sub_grid_index = level_offset + (index % self.UInt32(this_level_size))
        return self.n_features_per_level * sub_grid_index

    def __repr__(self):
        return (
            f"SimplifiedPermutohedral(\n"
            f"    dtype={self.dtype},\n"
            f"    dimension={self.dimension},\n"
            f"    hashmap_size={self.hashmap_size},\n"
            f"    n_levels={self.n_levels},\n"
            f"    n_features_per_level={self.n_features_per_level},\n"
            f"    base_resolution={self.base_resolution},\n"
            f"    per_level_scale={self.per_level_scale},\n"
            f"    smooth_weight_gradients={self.smooth_weight_gradients},\n"
            f"    smooth_weight_lambda={self.smooth_weight_lambda}\n"
            ")"
        )

