from __future__ import annotations
import sys
import drjit
import copy

if sys.version_info < (3, 11):
    from typing_extensions import Tuple, Type, overload, Iterable, List
else:
    from typing import Tuple, Type, overload, Iterable, List


def cosine_ramp(x: drjit.ArrayBase) -> drjit.ArrayBase:
    """ "Smoothed" ramp to help features blend-in without instabilities"""
    return 0.5 * (1.0 - drjit.cos(drjit.pi * x))


def div_round_up(num: int, divisor: int) -> int:
    """Compute ceiling division (num / divisor) using integer arithmetic."""
    return (num + divisor - 1) // divisor


def next_multiple(num: int, multiple: int) -> int:
    """Round `num` to the next multiple of `multiple`"""
    return multiple * div_round_up(num, multiple)


class HashEncoding:
    """
    This class serves as the interface for Hash based encodings.
    """

    # Fields, initialized by the ``alloc`` method.
    data: None | drjit.ArrayBase
    dtype: None | Type
    _level_offsets: None | List[int]

    # Fields, defining the hash encoding layout.
    _dimension: int
    _n_levels: int
    _n_features_per_level: int
    _hashmap_size: int
    _base_resolution: int
    _per_level_scale: float
    _log2_per_level_scale: float
    _align_corners: bool
    _torchngp_compat: bool
    _smooth_weight_gradients: bool
    _smooth_weight_lambda: float

    DRJIT_STRUCT = {
        "data": drjit.ArrayBase,
        "dtype": Type[drjit.ArrayBase],
        "_level_offsets": List[int],
        "_dimension": int,
        "_n_levels": int,
        "_n_features_per_level": int,
        "_hashmap_size": int,
        "_base_resolution": int,
        "_per_level_scale": float,
        "_align_corners": bool,
        "_torchngp_compat": bool,
        "_smooth_weight_gradients": bool,
        "_smooth_weight_lambda": float,
    }

    def __init__(
        self,
        dimension: int,
        *,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        hashmap_size: int = 2**19,
        base_resolution: int = 16,
        per_level_scale: float = 2,
        align_corners: bool = False,
        torchngp_compat: bool = False,
        smooth_weight_gradients: bool = False,
        smooth_weight_lambda: float = 1.0,
    ) -> None:
        self._dimension = dimension
        self._n_levels = n_levels
        self._n_features_per_level = n_features_per_level
        self._hashmap_size = hashmap_size
        self._per_level_scale = per_level_scale
        self._base_resolution = base_resolution
        self._log2_per_level_scale = drjit.log2(per_level_scale)
        self._align_corners = align_corners
        self._torchngp_compat = torchngp_compat
        self._smooth_weight_gradients = smooth_weight_gradients
        self._smooth_weight_lambda = smooth_weight_lambda

    def alloc(self, dtype: Type[drjit.ArrayBase], /) -> HashEncoding:
        """
        Allocates an Encoding with the specified type.
        Returns the new recording as a separate object instance.
        """

        result = copy.deepcopy(self)
        result._alloc(dtype)

        return result

    def _alloc(self, dtype: Type[drjit.ArrayBase]) -> None:
        """Initialize data storage and compute memory layout for hash grid levels.

        This function:
        1. Sets up DrJit array types based on the requested dtype
        2. Computes parameter counts per level, using dense grids for small scales
           and hash tables (limited by hashmap_size) for larger scales
        3. Calculates memory offsets for each level in the flattened data array
        4. Allocates the main data storage array
        5. Precomputes grid offset patterns for voxel corner indexing
        """
        self.dtype = drjit.leaf_t(dtype)
        self._init_types()

        # TODO: warning for n_features == 3

        assert (
            self.hashmap_size % 8
        ) == 0, f"Invalid hashmap size {self.hashmap_size}, must be a multiple of 8."

        self._level_offsets = [None] * (self.n_levels + 1)
        max_params = drjit.scalar.UInt32(0xFFFFFFFF) // 2
        offset = 0

        for level_i in range(self.n_levels):
            res = self._resolution(self._level_scale(level_i))
            stride = drjit.power(float(res), self.dimension)

            params_in_level = (
                int(max_params) if (stride > max_params) else (res**self.dimension)
            )
            params_in_level = next_multiple(params_in_level, 8)
            params_in_level = min(params_in_level, self.hashmap_size)

            self._level_offsets[level_i] = offset
            offset += params_in_level

        self._level_offsets[-1] = offset

        params_size = self._level_offsets[-1] * self.n_features_per_level
        self.data = drjit.zeros(self.dtype, params_size)

    def n_params(self) -> int:
        """
        The number of parameters, held by this encoding.
        """
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
        return self._dimension

    @property
    def hashmap_size(self) -> int:
        """
        The hashmap size provided when constructing this encoding.
        """
        return self._hashmap_size

    @property
    def n_levels(self) -> int:
        """
        The number of levels in this hash encoding.
        The actual number of output features of this encoding is determined by
        ``n_level * n_features_per_level``.
        """
        return self._n_levels

    @property
    def base_resolution(self) -> int:
        """
        The resolution of the 0th level.
        """
        return self._base_resolution

    @property
    def per_level_scale(self) -> float:
        """
        The per level scale factor, with which the scale of each level grows.
        """
        return self._per_level_scale

    @property
    def n_features_per_level(self) -> int:
        """
        The number of features per level.
        The actual number of output features of this encoding is determined by
        ``n_level * n_features_per_level``.
        """
        return self._n_features_per_level

    @property
    def align_corners(self) -> bool:
        """
        If the corners of the hashgrid should be aligned to the edges of its domain.
        """
        return self._align_corners

    @property
    def torchngp_compat(self) -> int:
        """Enable tiny-cuda-nn compatible indexing and offset calculations.

        When True, uses the same indexing functions, stride calculations, and
        position offsets as the reference tiny-cuda-nn implementation for
        reproducible results across implementations.
        """
        return self._torchngp_compat

    @property
    def smooth_weight_gradients(self) -> bool:
        """Whether to apply gradient smoothing to weights."""
        return self._smooth_weight_gradients

    @property
    def smooth_weight_lambda(self) -> float:
        """Blending factor for gradient smoothing using straight-through estimator.

        Controls the strength of the gradient smoothing applied to interpolation weights.
        A value of 1.0 fully replaces the original gradients with smoothed ones,
        while 0.0 disables smoothing entirely.
        """
        return self._smooth_weight_lambda

    @property
    def out_features(self) -> int:
        """Total dimensionality of the encoding's output feature vector.

        This is computed as n_levels * n_features_per_level, representing
        the concatenation of features from all resolution levels. For example,
        with 16 levels and 2 features per level, this returns 32.
        """
        return self._n_features_per_level * self.n_levels

    def _init_types(self):
        """
        Initializes the type aliases used for this hash encoding computation, given a
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

    def _position_types(
        self, p: drjit.ArrayBase
    ) -> Tuple[Type[drjit.ArrayBase], Type[drjit.ArrayBase]]:
        """
        Returns a tuple of the PositionFloat and PositionFloatXf types, given the
        position value given to the encoding.
        """
        mod = sys.modules[self.dtype.__module__]

        p = list(p)

        PositionFloat = drjit.leaf_t(p[0])

        PositionFloatXf = (
            mod.ArrayXf16 if drjit.is_half_v(PositionFloat) else mod.ArrayXf
        )

        return PositionFloat, PositionFloatXf

    def _acc_features(
        self,
        level_i: int,
        weight: drjit.ArrayBase,
        index: drjit.ArrayBase,
        values: List[drjit.ArrayBase],
        active: bool | drjit.ArrayBase,
    ):
        """
        Accumulates the ``self.num_features`` features into ``values`` at the given
        index. This function tries to use packet gather operations if possible,
        to improve the performance of the backward pass, as atomic packet scatters
        perform better than their non packeted counterparts for float16 types.
        """

        if self.smooth_weight_gradients:
            weight_smooth = cosine_ramp(weight)
            weight = weight + self.smooth_weight_lambda * (
                weight_smooth - drjit.detach(weight_smooth)
            )

        v = drjit.gather(
            self.StorageFloatXf,
            self.data,
            index,
            active,
            shape=(self.n_features_per_level, drjit.width(index)),
        )

        for k in range(0, self.n_features_per_level):
            values[level_i * self.n_features_per_level + k] = drjit.fma(
                v[k],
                self.StorageFloat(weight),
                values[level_i * self.n_features_per_level + k],
            )

    def indexing_function(self, key: drjit.ArrayBase, level_i: int) -> drjit.ArrayBase:
        """
        Given a key i.e. a D-dimensional integer vector, identifying a vertex
        of a simplex, this function calculates the index of the feature tuple,
        which is associated with that vertex.
        """

        scale = self._level_scale(level_i)
        res = self._resolution(scale)
        level_offset = self._level_offsets[level_i]
        this_level_size = self._level_offsets[level_i + 1] - level_offset

        # tiny-cuda-nn uses a uint32 here, which can lead to an integer overflow,
        # and the hash condition not being detected. For compatibility, we use
        # Dr.Jit's UInt32.
        stride = self.UInt32(1) if self.torchngp_compat else 1
        index = self.UInt32(0)
        for d in range(self.dimension):
            index += key[d] * self.UInt32(stride)
            stride *= res

            if stride > this_level_size:
                break

        # If the stride for dense indexing is too large, fallback to hash based indexing.
        if this_level_size < stride:
            index = self.hashing_function(key)

        sub_grid_index = level_offset + (index % self.UInt32(this_level_size))
        return sub_grid_index

    def hashing_function(self, key)->drjit.ArrayBase:
        """
        Hashes the D-dimensional key to compute a 1-dimensional index. This function
        is called when dense indexing is not possible.
        """
        raise NotImplementedError()

    def _level_scale(self, level: int) -> float:
        """
        This function computes the scale for a level of the hash encoding.

        The -1 means that `base_resolution` refers to the number of grid _vertices_ rather
        than the number of cells. This is slightly different from the notation in the paper,
        but results in nice, power-of-2-scaled parameter grids that fit better into cache lines.
        """
        return (
            drjit.exp2(level * self._log2_per_level_scale) * self.base_resolution
            - 1.0
        )

    def _resolution(self, scale: float) -> int:
        """
        Convert scale factor to discrete grid resolution (number of grid vertices).

        Adds 1 extra vertex when align_corners is False or torchngp_compat is True
        to account for different grid vertex alignment conventions.
        """
        return int(drjit.ceil(scale)) + (
            1 if not self.align_corners or self.torchngp_compat else 0
        )


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

    @overload
    def __init__(
        self,
        dimension: int,
        *,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        hashmap_size: int = 2**19,
        base_resolution: int = 16,
        per_level_scale: float = 2,
        align_corners: bool = False,
        torchngp_compat: bool = False,
        smooth_weight_gradients: bool = False,
        smooth_weight_lambda: float = 1.0,
    ) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(
        self, p: Iterable[drjit.ArrayBase], active: bool | drjit.ArrayBase = True
    ) -> Iterable[drjit.ArrayBase]:
        _, PositionFloatXf = self._position_types(p)

        p = PositionFloatXf(p)

        assert drjit.shape(p)[0] == self.dimension, (
            f"This hash grid expected an input of feature dimension {self.dimension}"
            f" but got {drjit.shape(p)[0]}."
        )

        grid_offsets = [
            self.ArrayXu([(i >> j) & 1 for j in range(self.dimension)])
            for i in range(2**self.dimension)
        ]

        out_values = [self.StorageFloat(0.0)] * (
            self.n_features_per_level * self.n_levels
        )

        for level_i in range(self.n_levels):
            scale = self._level_scale(level_i)

            p_offset: float = (
                0.5 if not self.align_corners or self.torchngp_compat else 0.0
            )
            pos = drjit.fma(p, scale, p_offset)
            pos0 = self.ArrayXu(drjit.floor(pos))

            w1 = pos - pos0
            w0 = 1.0 - w1

            for offset in grid_offsets:
                pos_grid = pos0 + offset
                weight = drjit.select(offset == 0, w0, w1)
                weight = drjit.prod(weight, axis=0)

                index = self.indexing_function(pos_grid, level_i)
                self._acc_features(level_i, weight, index, out_values, active)

        return self.StorageFloatXf(*out_values) & active

    def hashing_function(self, key):
        # Prime numbers used to hash the simplex vertices and calculate the lookup
        # index. These are equivalent to ones used for coherent hashing in tiny-cuda-nn.
        indexing_primes = [
            1,
            self.UInt32(2654435761),
            self.UInt32(805459861),
            self.UInt32(3674653429),
            self.UInt32(2097192037),
            self.UInt32(1434869437),
            self.UInt32(2165219737),
        ]
        index = self.UInt32(0)
        for d in range(0, self.dimension):
            index ^= key[d] * indexing_primes[d]

        return index

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

    def level_offset(self, level_i: int) -> int:
        """
        Helpful to build e.g. debug visualizations by splitting params per level.
        Must be called with an index up to and including `n_levels`.

        Warning: the level offset is expressed in number of vertices, i.e. it
        does *not* account for the feature count in each vertex. Each level contains
        `n_features_per_level` times the difference between the next offset entries.
        """
        return self._level_offsets[level_i]


class PermutoEncoding(HashEncoding):
    """
    Permutohedral lattice-based encoding inspired by :cite:`rosu2023permutosdf`.

    Unlike hash grid encodings that use regular grid lattices, this encoding employs
    a permutohedral lattice structure where simplices consist of triangles, tetrahedra,
    and higher-dimensional analogs. The key advantage is linear scaling: the number of
    vertices per simplex (and thus memory lookups per sample per level) grows linearly
    with dimensionality, compared to exponential growth in grid-based approaches.

    This implementation simplifies the original method by performing sorting and
    interpolation directly in d-dimensional space, avoiding the elevation to a
    hyperplane in (d+1)-dimensional space used in the reference implementation.

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

    @overload
    def __init__(
        self,
        dimension: int,
        *,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        hashmap_size: int = 2**19,
        base_resolution: int = 16,
        per_level_scale: float = 2,
        align_corners: bool = False,
        smooth_weight_gradients: bool = False,
        smooth_weight_lambda: float = 1.0,
    ) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(
        self, p: Iterable[drjit.ArrayBase], active: bool | drjit.ArrayBase = True
    ) -> Iterable[drjit.ArrayBase]:
        PositionFloat, PositionFloatXf = self._position_types(p)

        p = PositionFloatXf(p)

        assert drjit.shape(p)[0] == self.dimension, (
            f"This permutohedral grid expected an input of feature dimension {self.dimension}"
            f" but got {drjit.shape(p)[0]}."
        )

        out_values = [self.StorageFloat(0.0)] * (
            self.n_features_per_level * self.n_levels
        )

        for level_i in range(self.n_levels):
            scale = self._level_scale(level_i)

            # ---- Apply scaling factor

            p_offset: float = 0.0 if self.align_corners else 0.5
            x = drjit.fma(p, scale, p_offset)

            base = self.ArrayXi(drjit.floor(x))
            fract = x - base

            # ---- Insertion sort fract into xk and indices xki in ascending order
            xk = PositionFloatXf(fract)
            xki = self.ArrayXi([i for i in range(self.dimension)])

            for i in range(self.dimension):
                for j in range(i, self.dimension):
                    xk_i = PositionFloat(xk[i])
                    xk_j = PositionFloat(xk[j])
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
                PositionFloatXf, shape=(self.dimension + 1, drjit.width(p))
            )

            for rank in range(self.dimension):
                weights[rank] = xk[rank] - (xk[rank - 1] if rank > 0 else 0)

                # Calculates the vector from the ``base`` to the vertex
                # that is used for lookup.
                # We find the components of the offset vector, by iterating though
                # the sorted components ``xk``. For every iteration, we could set that
                # component of the offset vector to 1, for which ``fract >= xk[rank]``.
                # However, if two components of ``fract`` are equal i.e. the point
                # lies on the hyperplane between two simplexes, we have to decide
                # for one of them. We use the fact, that one of the equal components
                # is sorted before the other.

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
                self._acc_features(level_i, weight, index, out_values, active)

        return self.StorageFloatXf(*out_values) & active

    def hashing_function(self, key) -> drjit.ArrayBase:
        """Polynomial rolling hash for mapping lattice coordinates to hash table indices.

        Uses a simple multiplicative hash with prime number 2531011 to distribute
        lattice coordinates uniformly across the hash table space.
        """
        k = self.UInt32(0)
        for dim in range(self.dimension):
            k += key[dim]
            k *= 2531011
        return k

    def __repr__(self) -> str:
        return (
            f"PermutoEncoding(\n"
            f"    dtype={self.dtype},\n"
            f"    dimension={self.dimension},\n"
            f"    hashmap_size={self.hashmap_size},\n"
            f"    n_levels={self.n_levels},\n"
            f"    n_features_per_level={self.n_features_per_level},\n"
            f"    base_resolution={self.base_resolution},\n"
            f"    per_level_scale={self.per_level_scale},\n"
            f"    align_corners={self.align_corners},\n"
            f"    smooth_weight_gradients={self.smooth_weight_gradients},\n"
            f"    smooth_weight_lambda={self.smooth_weight_lambda}\n"
            ")"
        )
