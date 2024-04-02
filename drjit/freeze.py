# Copyright (c) 2024 NVIDIA CORPORATION.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from __future__ import annotations

from collections import namedtuple
from copy import deepcopy
import dataclasses
from enum import IntEnum
import functools
import inspect
import typing
from typing import List, Dict, Set, Tuple, Union, Type, Callable, Any, Final
from weakref import WeakSet

import drjit as _dr


ALLOWED_PYTHON_TYPES = (
    int,
    float,
    complex,
    bool,
    str,
    bytes,
    range,
    type(None),
    # TODO: nicer way to do this (but pybind11 enums are not subclasses of enum.Enum)
    _dr.ADMode,
    _dr.detail.ADScope,
    _dr.VarType,
    _dr.ADFlag,
    _dr.JitBackend,
    _dr.FilterMode,
    _dr.WrapMode,
    _dr.Scope,
    _dr.LogLevel,
    _dr.AllocType,
    _dr.ReduceOp,
    _dr.JitFlag,
    _dr.KernelType,
)


class ValueKind(IntEnum):
    # Depth-1 DrJit variable. Includes literals, as well as actual variables
    # that needs to be passed to the kernel as input or returned as output.
    VARIABLE = 0
    # Depth-2+ DrJit variable.
    # Internal-only, used to store info related to a composite array,
    # e.g. Array3f, before storing info about its components.
    COMPOSITE_ARRAY = 1
    # Simple Python-typed values (must remain constant across calls).
    PYTHON_VALUE = 2
    # Pointer to another JIT array
    POINTER = 3
    # Output buffers that need to be allocated before each launch. Typically,
    # the kernel would take a write-enabled pointer to these variables as
    # input, without seeing the variable itself.
    PREALLOCATED_OUTPUT = 4


FN_INPUT_PREFIX: Final[str] = "__fn_inputs"
FN_OUTPUT_PREFIX: Final[str] = "__fn_outputs"
KERNEL_OUTPUT_PREFIX: Final[str] = "__kernel_outputs"
USER_STATE_PREFIX: Final[str] = "__user_state_"
PREALLOCATED_OUTPUT_PREFIX: Final[str] = "__prealloc_output_"
GRAD_LITERALS_PREFIX: Final[str] = "__grad_literal"


# Key identifying a unique recorded kernel: (launch index, hash_low, hash_high)
KernelHash = Tuple[int, int, int]

# Hashable key that represents the path to the flattened variable.
#     kernel_hash: KernelHash or None
#     kind: ValueKind
#     path: str, dot-separated sequence of indices or keys leading to the value.
#     sub_path: str or None. If the value is a composite DrJit array, this field
#               can be used to index into the components (e.g. tuple of ints,
#               or '.grad'/'.value').
ArrayKey = namedtuple("ArrayKey", ("kernel_hash", "kind", "path", "sub_path"))

# Specifications for output buffers that we will need to pre-allocate before launches.
OutputBufferSpec = namedtuple(
    "OutputBufferSpec",
    ("py_type", "var_type", "original_width", "size_ratio", "match_width_of", "copy_from"),
)

# Used in cases where we need to record info about a DrJit flat array, but we don't hold
# the corresponding Python variable.
ArrayMock = namedtuple("ArrayMock", ("Type", "Depth", "IsDiff", "index", "size"))


def dr_var_type_to_py_type(vtype: _dr.VarType, package):
    a_name = _dr.detail.array_name(prefix="", vt=vtype, shape=(_dr.Dynamic,), scalar=False)
    return getattr(package, a_name)


class FlatVariables:
    def __init__(self):
        # ---------- Fields that must be filled in at each launch

        # Holds all of the variables (flat DrJit arrays, DrJit literals, pointers, Python
        # values, etc) encountered so far. The key indicates where this value originated from.
        self.flat_variables: Dict[ArrayKey, Any] = {}

        # Inverse mapping to quickly find the key corresponding to a DrJit variable.
        self.var_index_to_key: Dict[int, ArrayKey] = {}

        # ----------

        # ---------- Fields that are filled-in only once, at tracing time

        # Since variable indices are assigned in order, we record the index of a variable
        # created right after the evaluation of all input arguments, but after the execution
        # of the function. It will serve as a lower bound to determine whether an unknown
        # array (e.g. the target of a write-enabled pointer) was created inside of the function.
        self.index_lower_bound: int | None = None

        # For each *composite* DrJit variable, record the composite type so that we
        # can easily rebuild the function outputs in subsequent launches.
        # Types can be e.g. Float, Array3f, Matrix4f, etc.
        # We use the same key, with `ArrayKey.sub_path` = None to indicate this.
        self.dr_types: Dict[ArrayKey, Type] = {}

        # For each flattened DrJit variable, record whether gradients were enabled
        # before and the first function run. When replaying the frozen kernel, we will
        # check the "before" state matches, and enforce the "after" state.
        self.grad_enabled_before: Dict[ArrayKey, bool] = {}
        self.grad_enabled_after: Dict[ArrayKey, bool] = {}

        # Set of `ArrayKey` in `flat_variables` that correspond to literal variables
        # encountered in the function's inputs or outputs.
        # We need to hold on to these instances because:
        # - if a literal is passed as input, we must ensure that the same literal
        #   gets passed to later calls, because its value will be baked into the kernel.
        # - if a literal was created by the Python function and simply returned,
        #   it is constant w.r.t. the inputs, but we still need to be able to return it.
        self.literals: Set[ArrayKey] = set()

        # Set of `ArrayKey` in `flat_variables` that correspond to Python variables
        # encountered in the function's inputs or outputs.
        # If `check` mode is enabled, we keep the values passed in the first call and
        # verify that they never change in later call. This is important since
        # the frozen kernel will have all control flow and values from Python
        # baked based on the first call.
        # TODO: could allow Python-typed return values using this, although it's risky
        #       if any kind of Python state manages to sneak in but we silently ignore it.
        self.python_values: Set[ArrayKey] = set()

        # Set of `ArrayKey` in `flat_variables` that correspond to pre-allocated
        # output buffers that we will need to provide at kernel launch time.
        self.preallocated_outputs: Dict[ArrayKey, OutputBufferSpec] = {}

        # For each pointer variable: (ArrayKey for the source, is_input, is_write)
        self.pointers: Dict[ArrayKey, Tuple[ArrayKey, bool, bool]] = {}

        # Some function input variables may be replaced as a consequence of operations
        # within the function. For example a `dr.scatter()` may perform the scatter on
        # a copy of the target JIT variable, then the original Python/C++ array object
        # has its index replaced with the copy's index.
        # If we observe such a re-numbering at recording time, we record the source
        # and target paths here.
        self.renumbered_variables: Dict[ArrayKey, ArrayKey] = {}

        # ----------

    def __getitem__(self, index: Union[int, ArrayKey]):
        if isinstance(index, int):
            index = self.var_index_to_key[index]
        return self.flat_variables[index]

    def __contains__(self, index: Union[int, ArrayKey]):
        if isinstance(index, int):
            return index in self.var_index_to_key
        return index in self.flat_variables

    def __len__(self):
        return len(self.flat_variables)

    def copy_structure_shallow(self):
        """Returns a new instance of `FlatVariables`, with the non launch-specific fields
        pointing to this instances' values. Those fields should not be modified.
        Values that should remain available, e.g. literals, are added to the new
        instance's `flat_variable`s.
        """
        result = FlatVariables()
        for key in self.literals:
            arr = self.flat_variables[key]
            result.flat_variables[key] = arr
            result.var_index_to_key[arr.index] = key
        for key in self.python_values:
            result.flat_variables[key] = self.flat_variables[key]

        result.dr_types = self.dr_types
        result.grad_enabled_before = self.grad_enabled_before
        result.grad_enabled_after = self.grad_enabled_after
        result.literals = self.literals
        result.python_values = self.python_values
        result.preallocated_outputs = self.preallocated_outputs
        result.pointers = self.pointers
        result.renumbered_variables = self.renumbered_variables
        return result

    def get_dr_var(self, key: ArrayKey, is_diff: bool, aliases: Dict[ArrayKey, ArrayKey] = None):
        """
        Returns the depth 1 array at `key`, including its gradients array if it
        is a differentiable type and it had gradients enabled.
        Restores the grad_enabled status as well.
        """
        if is_diff:
            parent_key = key._replace(kind=ValueKind.COMPOSITE_ARRAY)
            array_type = self.dr_types[parent_key]

            sub_path = key.sub_path or ""
            sub_key1 = key._replace(sub_path=sub_path + ".value")
            assert array_type.Depth == 1

            if aliases is not None:
                assert (sub_key1 in self.literals) or (sub_key1 not in self.flat_variables)
                sub_key1 = aliases[sub_key1]
            value = self.flat_variables[sub_key1]

            result = array_type(value)
            if self.grad_enabled_after[parent_key]:
                sub_key2 = key._replace(sub_path=sub_path + ".grad")
                if aliases is not None:
                    assert (sub_key2 in self.literals) or (sub_key2 not in self.flat_variables)
                    sub_key2 = aliases[sub_key2]

                grad = self.flat_variables[sub_key2]

                # TODO: we need a general solution for literals with width > 1 at launch time
                if sub_key2 in self.literals and _dr.width(grad) != _dr.width(result):
                    _dr.resize(grad, _dr.width(result))

                _dr.set_grad_enabled(result, True)
                result.set_grad_(grad)
        else:
            array_type = self.dr_types[key]
            if aliases is not None:
                assert (key in self.literals) or (key not in self.flat_variables)
                key = aliases[key]
            # assert self.dr_types[key].Depth == 1
            result = array_type(self.flat_variables[key])

        return result

    def add_dr_var(
        self,
        aliases: Dict[ArrayKey, ArrayKey],
        parent_key: ArrayKey,
        parent_dr_type: Type,
        v: _dr.ArrayBase,
        sub_path: Any = None,
        record: bool = False,
        must_find_alias: bool = False,
        check: bool = False,
    ) -> ArrayKey:
        # By the time we call this method, all "composite" types should have been
        # broken down. We just handle diff / not diff variables.
        assert v.Depth == 1

        if is_really_diff_v(v):

            sub_path_ = sub_path or ""
            key1 = self.add_flat_dr_var(
                aliases,
                parent_key,
                parent_dr_type,
                v.detach_ref_(),
                sub_path=sub_path_ + ".value",
                record=record,
                must_find_alias=must_find_alias,
                check=check,
            )
            key2 = self.add_flat_dr_var(
                aliases,
                parent_key,
                parent_dr_type,
                v.grad_(),
                sub_path=sub_path_ + ".grad",
                record=record,
                must_find_alias=must_find_alias,
                check=check,
            )

            # Record type and grad_enabled for the Diff type of this sub-component,
            # e.g. `cuda.ad.Array3f.x`, not the value-only `cuda.ad.Array3f.x.detach_ref_()`
            v_grad_enabled = v.grad_enabled_()
            # if v_grad_enabled:
            #     v_grad = _dr.grad(v)
            #     print(f'{parent_key=} --> {v_grad_enabled=}, {v_grad.index=}, {is_literal(v_grad)=}, grad_key={key2}')

            composite_key = key1._replace(sub_path=sub_path, kind=ValueKind.COMPOSITE_ARRAY)
            if record:
                self.grad_enabled_before[composite_key] = v_grad_enabled
                # Let's assume it won't by default. This will be checked for input
                # variables in `record_inputs_grads`.
                self.grad_enabled_after[composite_key] = self.grad_enabled_before[composite_key]

            elif self.grad_enabled_before[composite_key] != v_grad_enabled:
                raise ValueError(
                    f"When recording the function, the array at {composite_key.path + (composite_key.sub_path or '')}"
                    f" had grad_enabled={self.grad_enabled_before[composite_key]}, which doesn't match"
                    f" the current grad_enabled={v_grad_enabled}. This is not supported."
                )

            if composite_key in self.dr_types:
                assert self.dr_types[composite_key] == type(v)
            else:
                self.dr_types[composite_key] = type(v)

            return (key1, key2)
        else:
            return self.add_flat_dr_var(
                aliases,
                parent_key,
                parent_dr_type,
                v,
                sub_path=sub_path,
                record=record,
                must_find_alias=must_find_alias,
                check=check,
            )

    def add_flat_dr_var(
        self,
        aliases: Dict[ArrayKey, ArrayKey],
        parent_key: ArrayKey,
        parent_dr_type: Type,
        v: _dr.ArrayBase,
        sub_path: Any = None,
        record: bool = False,
        must_find_alias: bool = False,
        check: bool = False,
    ) -> ArrayKey:
        # By the time we call this method, all "composite" and differentiable types
        # should have been broken down.
        assert v.Depth == 1 and not is_really_diff_v(v)

        is_lit = is_literal(v)
        key = parent_key._replace(kind=ValueKind.VARIABLE, sub_path=sub_path)

        if record:
            if parent_key in self.dr_types:
                assert self.dr_types[parent_key] == parent_dr_type
            else:
                self.dr_types[parent_key] = parent_dr_type

            assert (key not in self.dr_types) or (parent_key == key)
            self.dr_types[key] = type(v)
        else:
            if parent_key not in self.dr_types:
                raise ValueError(
                    "Error parsing frozen function inputs."
                    " When generating the kernel during the function's first run,"
                    f' parameter "{parent_key.path}{parent_key.sub_path or ""}" was not a DrJit array.'
                    f" However, in this function call, it has type {parent_dr_type},"
                    " which is not supported."
                )

            if is_lit != (key in self.literals):
                raise ValueError(
                    "Error parsing frozen function inputs."
                    " When generating the kernel during the function's first run,"
                    f' parameter "{parent_key.path}{parent_key.sub_path or ""}" was'
                    f' a {"non " if is_lit else ""}literal-valued'
                    f" DrJit array. However, in this function call, it has type {parent_dr_type},"
                    f' and is {"" if is_lit else "not "}a literal. This is not supported.'
                )

            if check and self.dr_types[parent_key] != parent_dr_type:
                raise ValueError(
                    "Error parsing frozen function inputs."
                    " When generating the kernel during the function's first run,"
                    f' parameter "{parent_key.path}{parent_key.sub_path or ""}"'
                    f" was a DrJit array with type {self.dr_types[parent_key]}."
                    f" However, in this function call, it has type {parent_dr_type},"
                    " which is not supported."
                )

        if not record and is_lit:
            # Literal variables are kept from the first run,
            # so we expect to have it already.
            assert key in self.flat_variables
        else:
            assert key not in self.flat_variables
            self.flat_variables[key] = v

        if v.index in self.var_index_to_key:
            # This variable is present twice in the inputs (aliasing)
            if record:
                aliases[key] = self.var_index_to_key[v.index]

            # TODO: is it a problem is a value did not alias at recording time but aliases now?
            # if not record and key not in aliases:
            #     raise ValueError()

        else:
            if not record and key in aliases:
                alias = aliases[key]
                sub1 = f"{key.sub_path}" if key.sub_path else ""
                sub2 = f"{alias.sub_path}" if alias.sub_path else ""
                raise ValueError(
                    "Error preparing kernel inputs from the function arguments."
                    " When generating the kernel during the function's first run,"
                    f' parameter "{key.path}{sub1}" aliased with the other input'
                    f' "{alias.path}{sub2}", and therefore resulted only in a'
                    " single input to the kernels. However, in this function call, these two"
                    " inputs no longer alias and are therefore not compatible with the"
                    " frozen kernel."
                )

            self.var_index_to_key[v.index] = key

        if is_lit:
            if record:
                # TODO: there are some literals that we don't need to hold on to if we're not
                #       in `check` mode (e.g. function inputs that are not part of the outputs).
                self.literals.add(key)

                # It's possible that the Python function returns a literal
                # that "comes out of nowhere" (i.e. simply created in the function).
                # In that case, we record a self-alias here so that the checks later
                # are satisfied.
                if must_find_alias and key not in aliases:
                    aliases[key] = key

            else:
                if key not in self.literals:
                    raise ValueError(
                        "Error preparing kernel inputs from the function arguments."
                        " When generating the kernel during the function's first run,"
                        f' parameter "{key.path}{key.sub_path or ""}" was a'
                        f" literal-valued DrJit array."
                        f" However, in this function call, it has type {type(v)},"
                        " and is not a literal. This is not supported."
                    )

                if check:
                    ref = self.flat_variables[key]
                    if (type(v) != type(ref)) or _dr.any(_dr.neq(v, ref)):
                        raise ValueError(
                            "Error preparing kernel inputs from the function arguments."
                            " When generating the kernel during the function's first run,"
                            f' argument "{key.path}{key.sub_path or ""}" had literal'
                            f" value {ref}(type: {type(ref)}), which was baked into the kernel."
                            f" However, in this function call, a different value ({v},"
                            f" type {type(v)}) was given to this argument, which will lead"
                            " to incorrect results."
                        )

        if must_find_alias and key not in aliases:
            # We flattened the function outputs and added them to `all_variables`.
            # Assumings that none of these function outputs appeared out of nowhere,
            # all of the DrJit-typed values should have already been
            # present in `all_variables`. We use `self.aliases` to find the key
            # under which the value we are looking for was actuall stored.
            raise ValueError(
                f"Error parsing function outputs: DrJit-typed return value at {key=}"
                f" with index={v.index}, type={type(v)} was not found in any of the"
                " kernel outputs or initial function inputs."
            )

        return key

    def add_python_value(self, path: str, v: ALLOWED_PYTHON_TYPES) -> ArrayKey:
        # Not applicable, only the Python function may take or return Python values
        kernel_hash = None

        key = ArrayKey(kernel_hash, ValueKind.PYTHON_VALUE, path, sub_path=None)
        assert key not in self.flat_variables

        self.flat_variables[key] = v
        self.python_values.add(key)

        return key

    def add_pointer(
        self,
        kernel: FrozenKernel,
        path: str,
        is_parsing_inputs: bool,
        unclaimed_arrays: Dict[int, _dr.ArrayBase],
        ptr_info: _dr.PointerInfo,
    ) -> ArrayKey:
        write_enabled = ptr_info.write_enabled
        pointer_index = ptr_info.pointer_index
        source_index = ptr_info.source_index

        if write_enabled and not ptr_info.is_input:
            # TODO: we should probably be able to support this case, it's not a big deal?
            raise ValueError(
                "Error parsing frozen kernel outputs: found write-enabled pointer"
                f" variable (index {pointer_index}, pointing variable {source_index})"
                " in the kernel outputs, which is not supported."
            )

        # --- Detect write-enabled pointers to existing (input) or created (output) arrays.
        # We will need to pre-allocate those buffers if they were not part of the
        # Python function inputs.
        source_array_from_unclaimed = unclaimed_arrays.get(source_index)
        if (
            is_parsing_inputs
            and write_enabled
            and ((source_array_from_unclaimed is not None) or (source_index > self.index_lower_bound))
            and (source_index not in self.var_index_to_key)
        ):
            # TODO: there should be no other pointers to that buffer outside of the frozen
            #       function, and the variable should not be read from or used in any
            #       other way outside either. Otherwise, there is a risk that the
            #       function wanted to write in-place to an existing array *and*
            #       return a reference to it.
            #       If we don't detect this case, we could accidentally replace it with a
            #       new output buffer and lose the side-effect.
            if source_array_from_unclaimed is not None:
                output_array = unclaimed_arrays[source_index]
                assert output_array.Depth == 1, "Pointers should point to a single flat array"
                assert not is_really_diff_v(output_array), "Pointers should point to a single flat array"
            else:
                # We don't have a handle on the array, but we know it was allocated within the
                # function and we have enough information to record it.
                output_array = ArrayMock(
                    Type=ptr_info.source_type, Depth=1, IsDiff=False, index=source_index, size=ptr_info.source_size
                )

            # Note: here we record the pre-allocated buffer itself, not the pointer to it.
            key = ArrayKey(
                kernel.hash,
                ValueKind.PREALLOCATED_OUTPUT,
                PREALLOCATED_OUTPUT_PREFIX + path,
                sub_path=None,
            )
            self.flat_variables[key] = output_array
            self.var_index_to_key[source_index] = key
            if isinstance(output_array, ArrayMock):
                py_type = dr_var_type_to_py_type(output_array.Type, _dr.cuda)
                original_width = output_array.size
            else:
                py_type = type(output_array)
                original_width = _dr.width(output_array)

            self.preallocated_outputs[key] = OutputBufferSpec(
                py_type=py_type,
                var_type=output_array.Type,
                original_width=original_width,
                size_ratio=None,  # Will be filled-in later
                match_width_of=None,  # Will be filled-in later
                copy_from=None,  # Will be filled-in later if appropriate
            )

        # --- Record the actual pointer variable
        if source_index in self.var_index_to_key:
            # Read or write pointer to an existing variable.
            # Note: even though the pointer itself may be considered as an input from
            # the point of view of the kernel (`is_input`), it may be pointing to an
            # output variable (in that case, we expect `write_enabled == True`).

            if ptr_info.is_input != is_parsing_inputs:
                # Consider only input (resp. output) variables as appropriate
                return

            # TODO: we should probably be able to support this case, it's not a big deal?
            if not is_parsing_inputs and (pointer_index in self.var_index_to_key):
                raise NotImplementedError(
                    "Not supported yet: passing a pointer variable from the inputs to the outputs"
                )

            # Note: we assume that the source var was already encountered when
            # flattening the inputs, so we don't update `self.width` here.
            source_key = self.var_index_to_key[source_index]
            # Cannot support DrJit pointer variables to literals or Python values
            assert source_key.kind in (
                ValueKind.VARIABLE,
                ValueKind.PREALLOCATED_OUTPUT,
            )

            pointer_key = ArrayKey(kernel.hash, ValueKind.POINTER, path, sub_path=None)
            # It's probably not so important to hold a reference to the actual pointer variable,
            # so we're not going to try and get it. Just set a None entry as placeholder.
            self.flat_variables[pointer_key] = None
            self.pointers[pointer_key] = (source_key, ptr_info.is_input, write_enabled)
            self.var_index_to_key[pointer_index] = pointer_key

        else:
            # TODO: is it's not write-enabled, we have a message saying to make it
            # part of the user-state, and I think we shouldn't get to this place in the code?
            rw_str = "write-enabled" if write_enabled else "read-only"
            raise ValueError(
                f"Error parsing frozen kernel inputs: found {rw_str} pointer"
                f" variable (index {pointer_index}, pointing to var {source_index}),"
                f" but we could not find variable {source_index}. It may be because"
                " the target array was allocated outside of the frozen function, but was"
                " not given as part of the `state` lambda (argument of @dr.kernel())."
            )

    def add_kernel_output(
        self,
        kernel: FrozenKernel,
        var_index: int,
        tp: _dr.VarType,
        size: int,
        path: str,
    ):
        """Add a reference to a flat DrJit array that was output by a kernel.
        Since we do not hold the actual variable corresponding to this array, we can only
        record a mock for it.
        """
        if tp == _dr.VarType.Pointer:
            raise NotImplementedError("Not supported yet: pointer-typed outputs from frozen kernels.")

        key = ArrayKey(kernel.hash, ValueKind.VARIABLE, path, sub_path=None)
        assert var_index not in self.var_index_to_key
        self.var_index_to_key[var_index] = key
        # Note: ArrayMock doesn't work with `dr.width()` and so on.
        self.flat_variables[key] = ArrayMock(index=var_index, Type=tp, Depth=1, IsDiff=False, size=size)

    def add_kernel_output_real(self, kernel: FrozenKernel, arr: _dr.ArrayBase, path: str):
        """Same as `add_kernel_output`, except that at launch time we have the actual
        variable for the kernel output.
        """
        if arr.Type == _dr.VarType.Pointer:
            raise NotImplementedError("Not supported yet: pointer-typed outputs from frozen kernels.")

        # We expect only plain flat types as kernel outputs
        assert arr.Depth == 1 and not is_really_diff_v(arr)

        key = ArrayKey(kernel.hash, ValueKind.VARIABLE, path, sub_path=None)
        assert arr.index not in self.var_index_to_key
        self.var_index_to_key[arr.index] = key
        self.flat_variables[key] = arr

    def record_renumbered_variables(self):
        """
        Assuming this instance contains all function inputs, which were recorded
        *before* the function evaluation, some of the variables may have been
        re-numbered in-place by the function.
        Detect those cases.
        """
        for old_index, key in self.var_index_to_key.items():
            if not key.path.startswith(FN_INPUT_PREFIX) or key.kind != ValueKind.VARIABLE:
                continue

            # Note: for AD vars, we stored `v.detach_ref_` and `v.grad_()` separately,
            # but not `v`. Unfortunately, `v.grad_()` returns a fresh Python variable
            # that has the *current* AD index of `v`. If the AD index happens to
            # be replaced in `v`, it won't be updated in the gradient variable that
            # we stored.
            # However, since the grad variable should only be manipulated by the
            # AD-related features such as gradient accumulation, the AD variable
            # index should remain constant (?).

            new_index = self.flat_variables[key].index
            if old_index == new_index:
                # No renumbering, nothing to do for that input
                continue

            # At this point we know that `self.var_index_to_key[old_index]`
            # points to the renumbered variable, which doesn't have the same
            # index anymore.
            # TODO: do we need this entry anywhere else?
            # del self.var_index_to_key[old_index]

            # There was some renumbering, try to indentify what the new index
            # corresponds to. Note that pre-allocated buffers should already have
            # been detected at this point.
            if new_index not in self.var_index_to_key:
                raise KeyError(
                    f"Function input {key} (index r{old_index}) was renumbered to r{new_index},"
                    " but we could not find what this new variable index corresponds to."
                )

            new_key = self.var_index_to_key[new_index]
            self.renumbered_variables[key] = new_key

            if new_key in self.preallocated_outputs:
                value = self.flat_variables[key]
                spec = self.preallocated_outputs[new_key]
                if spec.var_type != value.Type or spec.original_width != _dr.width(value):
                    raise ValueError(
                        f"After detecting that function input {key} (index r{old_index}) was"
                        f" renumbered to r{new_index}, we found that r{new_index} should be a newly"
                        f" allocated copy of r{old_index}."
                        f" However, the types ({spec.var_type} vs {value.Type}) or"
                        f" or width ({spec.original_width} vs {_dr.width(value)}) didn't match."
                    )
                self.preallocated_outputs[new_key] = spec._replace(copy_from=key)

    def get_unclaimed_arrays(self, history: List[Dict]) -> Dict[int, _dr.ArrayBase]:
        """
        Returns a dict (var index -> value) of the DrJit arrays that were
        recorded in this instance, but that do not appear in the input
        or output variables of any kernel in `history`.

        Although they do not appear in the direct inputs or outputs of a kernel,
        they may well:
        - Have been created in the function and returned
        - Be the target of a read- or write-enabled pointer that *is* an input to the kernel.

        We also include arrays that are first refered to via a write-enabled pointer,
        and then in a later kernel used as input or via a read-enabled pointer. However,
        we do *not* want to include arrays that are directly read from or only written to,
        since those cases are more likely to be pre-existing arrays that the user forgot
        to include in the `state` lambda.
        """

        # Arrays that appear in the function inputs or outputs, but not as direct
        # input or outputs of individual kernels.
        claimed = set()
        for entry in history:
            claimed.update([p[0] for p in entry["input_variables"]])
            claimed.update([p[0] for p in entry["output_variables"]])

        result = {}
        for key, value in self.flat_variables.items():
            if key.kind == ValueKind.PYTHON_VALUE:
                continue
            var_index = value.index
            if var_index in claimed:
                continue
            result[var_index] = value

        # Arrays that are first written to, and then used.
        # Since we don't actually have access to the array itself, we
        # create a proxy so that the rest of the frozen kernel tracing
        # logic can act on it.
        kernel_for_read = {}
        kernel_for_unclaimed_write = {}
        for kernel_i, entry in enumerate(history):
            for index, _, _ in entry["input_variables"]:
                kernel_for_read.setdefault(index, kernel_i)

            for ptr_info in entry["pointer_variables"]:
                if ptr_info.source_index in self.var_index_to_key:
                    # Pointer to a known array, that's not the case we are trying to handle here.
                    continue
                if ptr_info.write_enabled:
                    kernel_for_unclaimed_write.setdefault(ptr_info.source_index, (kernel_i, ptr_info))
                else:
                    kernel_for_read.setdefault(ptr_info.source_index, kernel_i)

        for var_index, (wrote_in_kernel_i, ptr_info) in kernel_for_unclaimed_write.items():
            read_in_kernel_i = kernel_for_read.get(var_index)
            if read_in_kernel_i is None:
                # Really points to an unknown buffer, user probably forgot to include
                # it in the state lambda.
                continue
            if wrote_in_kernel_i > read_in_kernel_i:
                # Started using the variable before it was written to, it might still be
                # a pre-existing array that the user forgot to include in `state`.
                # TODO: should we allow this nonetheless?
                continue

            result[var_index] = ArrayMock(
                Type=ptr_info.source_type, Depth=1, IsDiff=False, index=var_index, size=ptr_info.source_size
            )

        return result

    def preallocate_buffers(self, kernel: FrozenKernel, max_width: int):
        for key, spec in self.preallocated_outputs.items():
            if key.kernel_hash != kernel.hash:
                continue

            assert (
                max_width > 0
            ), "Not supported: guessing the size of a pre-allocated output buffer for a kernel with no inputs."
            # The size of the output buffer is hopefully a multiple of the launch size
            if spec.size_ratio is not None:
                is_multiple, factor = spec.size_ratio
                sz = apply_factor_or_ratio(max_width, is_multiple, factor)
            else:
                assert spec.match_width_of is not None
                if spec.match_width_of not in self.flat_variables:
                    raise KeyError(
                        "When first tracing the frozen function, we recorded a new buffer allocation"
                        f" ({key=}) whose width matched that of {spec.match_width_of}."
                        f" However, we could not locate that variable in this subsequent launch"
                        " in order to determine the correct buffer size to allocate."
                    )
                sz = _dr.width(self.flat_variables[spec.match_width_of])

            if spec.copy_from is not None:
                source = self.flat_variables[spec.copy_from]
                assert _dr.width(source) == sz
                assert source.Type == spec.var_type
                array = source.copy_()
            else:
                # TODO: how can we guess what initialization the first run used in order to match it?
                array = _dr.opaque(spec.py_type, 0, sz)

            _dr.schedule(array)

            assert key not in self.flat_variables
            self.var_index_to_key[array.index] = key
            self.flat_variables[key] = array

    def create_pointer_variables(self, kernel: FrozenKernel):
        for key, (source_key, is_input, write_enabled) in self.pointers.items():
            # We only create pointers that will be consummed by this kernel, because
            # some source variables may only become available later.
            if key.kernel_hash != kernel.hash:
                continue

            if source_key not in self.flat_variables:
                raise KeyError(
                    "Error when preparing kernel inputs for a frozen launch:"
                    f" pointer variable {key} ({write_enabled=}) should point to"
                    f" source variable {source_key}, but that variable is not available."
                )

            # TODO: why should there not be LVN? We do observe multiple identical
            #       pointer variables in the generated kernels, but why is that?
            # TODO: this could trigger an eval before we have scheduled all of the
            #       inputs, which could be wasteful (single launch) or even incorrect
            #       (multiple launches)?
            source_value = self.flat_variables[source_key]
            if source_value.IsDiff:
                # TODO: should probably have a `source_key` that points directly
                #       to the detached value instead? It's probably a problen of
                #       mismatch between `is_diff_v` and `is_really_diff_v`, e.g. for
                #       `dr.cuda.ad.UInt32`.
                source_value = source_value.detach_ref_()
            ptr_value = source_value.data_var(write_enabled, disable_lvn=True)

            assert key not in self.flat_variables
            self.flat_variables[key] = ptr_value

    def remove_pointer_variables(self, kernel: FrozenKernel):
        for key, (_, is_write, _) in self.pointers.items():
            if not is_write or (key.kernel_hash != kernel.hash):
                continue

            # Let's hope these same pointer isn't used later on?
            self.flat_variables[key].reset()
            del self.flat_variables[key]

    def renumber_inputs(self, kwargs: Dict[str, Any]):
        if not self.renumbered_variables:
            return

        # For each variable to renumber, need to find the Python variable
        # in the `kwargs` (which may be deeply nested + composite) and
        # renumber it in-place to the newly allocated variable.
        for old_key, new_key in self.renumbered_variables.items():
            assert old_key.kind == ValueKind.VARIABLE
            entry = get_array_ref_for_key(kwargs, old_key, path_prefix=FN_INPUT_PREFIX)

            new_value = self.flat_variables[new_key]
            entry.set_index_ref_counted_(new_value.index)


def is_literal(v):
    return v.Type != _dr.VarType.Pointer and v.is_literal_()


def is_really_diff_v(v):
    return v.IsDiff and v.IsFloat


def size_ratio(size1: int, size2: int) -> Tuple[bool, int, bool]:
    # Returns: (is_multiplier, factor, is_valid)
    if size2 >= size1:
        size1 = max(size1, 1)
        valid = (size2 % size1) == 0
        return (False, size2 // size1, valid)
    else:
        size2 = max(size2, 1)
        valid = (size1 % size2) == 0
        return (True, size1 // size2, valid)


def apply_factor_or_ratio(x: int, is_factor: bool, factor: int) -> int:
    x = max(x, 1)
    if is_factor:
        return x * factor
    else:
        return x // factor


def get_array_ref_for_key(obj: Any, key: ArrayKey, path_prefix: str = None):
    # We assume that '.' is not a valid character in kwargs' names
    path = key.path.split(".")
    if path_prefix:
        assert path[0] == path_prefix
        path = path[1:]
    entry = get_array_for_path(obj, path)
    if key.sub_path is not None:
        entry = get_subarray_ref_for_path(entry, key.sub_path.split("."))
    return entry


def get_array_for_path(obj: Any, path: list[str]):
    entry = obj
    for p in path:
        # Walk down through the hierarchy to locate the variable at `path`
        if isinstance(entry, dict):
            entry = entry[p]
        elif isinstance(entry, (tuple, list)):
            entry = entry[int(p)]
        elif dataclasses.is_dataclass(entry) or _dr.is_struct_v(entry):
            entry = getattr(entry, p)
        else:
            raise KeyError(f"Could not find key {path} in object: {obj}")

    if not _dr.is_jit_v(entry):
        raise ValueError(
            f"Entry at path {path} of kwargs was expected to be"
            f" a DrJit array, but found type = {type(entry)}, value = {entry}"
        )

    return entry


def get_subarray_ref_for_path(array: _dr.ArrayBase, sub_path: list[str]):
    depth = _dr.depth_v(array)
    assert sub_path[0] == ""  # Starts with a '.', if any
    assert len(sub_path) in range(1, 5), f"Unexpected: {sub_path=}"

    if depth == 2:
        array = array.entry_ref_(int(sub_path[1]))
    elif depth == 3:
        array = array.entry_ref_(int(sub_path[1])).entry_ref_(int(sub_path[2]))
    elif depth > 3:
        raise NotImplementedError("Not supported yet: Depth > 3")

    if sub_path[-1] == "value":
        array = array.detach_ref_()
    elif sub_path[-1] == "grad":
        array = array.grad_()

    return array


class KernelInputsMapping:
    def __init__(self):
        # Mapping from the internal ArrayKey to the position in the inputs (resp.
        # outputs) of the kernel.
        # This is used in subsequent launches to rebuild the outputs to be passed
        # to the kernels from e.g. the function outputs and previous kernels' outputs.
        # Note that there may still be variables passed as input to the function
        # that are simply not used by the kernel, in which case the value is None.
        self.key_to_kernel_idx: Dict[ArrayKey, int] = {}
        self.kernel_idx_to_key: List[ArrayKey] = {}

        # Maximum size of the variables actually passed to the kernel
        self.max_width: int = 0

    def record_kernel(
        self,
        all_variables: FlatVariables,
        kernel: FrozenKernel,
        is_parsing_inputs: bool,
        unclaimed_arrays: Dict[int, _dr.ArrayBase],
    ):
        # Record pointers and pre-allocated output buffers, if any.
        # To help differentiate cases where we _should_ allocate a scratch or return
        # buffer vs the cases where the user forgot to pass an existing buffer as part
        # of the state lambda, we start by registering the write-enabled pointers.
        pointer_variables = sorted(
            kernel.history_entry["pointer_variables"], key=(lambda ptr_info: 0 if ptr_info.write_enabled else 1)
        )

        for i, ptr_info in enumerate(pointer_variables):
            # TODO: if a pre-allocated array is detected, do we need to record its key
            #       here? Or is it enough that `all_variables` knows about it?
            path = f"__pointer_{i}"
            _ = all_variables.add_pointer(kernel, path, is_parsing_inputs, unclaimed_arrays, ptr_info)

            self.max_width = max(self.max_width, _dr.width(all_variables[ptr_info.source_index]))

        # Identify which of `all_variables` participate in the kernel and record
        # their slot positions.
        indices_observed = kernel.history_entry["input_variables" if is_parsing_inputs else "output_variables"]
        assert len(all_variables) >= len(indices_observed)
        # Convert to dict (variable index -> kernel param index) for faster lookup.
        indices_observed = {idx: i for i, (idx, _, _) in enumerate(indices_observed)}

        self.kernel_idx_to_key = []
        for i, var_idx in enumerate(indices_observed):
            if var_idx not in all_variables:
                raise KeyError(
                    f'Error parsing kernel {"inputs" if is_parsing_inputs else "outputs"}:'
                    f" variable index {var_idx} was in the {i}th kernel slot, but we could not"
                    " find it in all of the variables encountered so far."
                    " If the function accesses some existing variables that are not part of the"
                    " function arguments, they must be passed explicitly via the `state` argument"
                    " of the `dr.kernel()` decorator."
                    " This may also happen when an array is created within the function, including"
                    " by using dr.opaque()."
                )
            key = all_variables.var_index_to_key[var_idx]
            self.key_to_kernel_idx[key] = i
            self.kernel_idx_to_key.append(key)
            self.max_width = max(self.max_width, _dr.width(all_variables[key]))

        # If any pre-allocated buffer are to be created for this kernel,
        # update their original size ratio
        for key in all_variables.preallocated_outputs:
            if key.kernel_hash == kernel.hash:
                desc = all_variables.preallocated_outputs[key]
                is_multiplier, factor, is_valid = size_ratio(desc.original_width, kernel.original_launch_size)
                # Since `size_ratio` could easily accidentally work for small launch sizes,
                # so we prefer using `match_width_of` in those cases.
                if (kernel.original_launch_size > 2) and is_valid:
                    # The width of the buffer looks like a simple multiple or
                    # ratio of the launch size, we'll use the same factor in
                    # subsequent launches.
                    desc = desc._replace(size_ratio=(is_multiplier, factor))
                else:
                    # Try to identify a variable whose width matches this one.
                    # This could happen e.g. when allocated gradient buffers.
                    # TODO: this might break down for width-1 arrays?
                    ref_key = None
                    for other_key, other_val in all_variables.flat_variables.items():
                        if _dr.width(other_val) == desc.original_width:
                            ref_key = other_key
                            break

                    if ref_key is not None:
                        desc = desc._replace(match_width_of=ref_key)
                    else:
                        raise ValueError(
                            f"FrozenKernel: one of the function outputs ({key=}) is a preallocated array,"
                            " which the kernel writes to (probably via a `scatter` operation)."
                            " However, we could not guess an appropriate allocation size based"
                            f" on the original buffer size ({desc.original_width})"
                            f" and launch size ({kernel.original_launch_size}). We also could not"
                            " find a variable whose width matches."
                        )

                all_variables.preallocated_outputs[key] = desc


class FunctionInputsMapping:
    """
    Helper class to construct the mapping from Python function inputs
    to kernel inputs, or from kernel outputs to Python function outputs.

    Handles cases such as:
    - Arrays that are literals, and therefore not recorded as an input to the kernel
    - Outputs that were part of the inputs and just passed as-is
    - Tuple-, list- and dict-typed inputs and outputs (not nested)
    """

    def __init__(self):
        # Mapping from the path constructed in traversal order (e.g. using list indices
        # and dict keys) to `ArrayKey`s used to index into collected `FlatVariables`.
        # This helps map high-level arrays, e.g. from the Python function inputs,
        # to their flattened components.
        self.path_to_key: Dict[str, Union[ArrayKey, List[ArrayKey], List[List[ArrayKey]]]] = {}

        # Records which key aliases to which other key. It's important that
        # the aliasing in subsequent launches matches the one at record time,
        # otherwise we may silently ignore inputs that should now be different.
        self.aliases: Dict[ArrayKey, ArrayKey] = {}

    def flatten_all(
        self,
        arrays,
        all_variables: FlatVariables,
        path: str,
        is_parsing_inputs: bool,
        is_recording: bool,
        must_find_alias: bool = False,
        check: bool = False,
        path_to_type: Dict[str, Tuple(Type, bool, str)] | None = None,
        parent_path: str = "",
    ):
        """
        Flattens the input array recursively. The following is introduced to reconstruct
        nested types, namely tuple, list, dict or DRJIT_STRUCT:
        - we can optionnally record the path of the parent when calling flatten_all recursively
        - `path_to_type` stores the (type, nestedness, parent_path) at each node of the
          flattening tree.
        """
        if not is_recording:
            # At launch time, we don't have access to overall function outputs (it's our job
            # to build them), so this function will never be called on outputs except at
            # recording time.
            assert is_parsing_inputs

        if _dr.is_jit_v(arrays):
            depth = _dr.depth_v(arrays)
            dr_type = type(arrays)
            parent_key = ArrayKey(None, kind=ValueKind.COMPOSITE_ARRAY, path=path, sub_path=None)
            self.path_to_key[path] = parent_key

            if depth == 1:
                all_variables.add_dr_var(
                    self.aliases,
                    parent_key,
                    dr_type,
                    arrays,
                    record=is_recording,
                    must_find_alias=must_find_alias,
                    check=check,
                )

            elif depth == 2:
                for k in range(arrays.Size):
                    all_variables.add_dr_var(
                        self.aliases,
                        parent_key,
                        dr_type,
                        arrays.entry_ref_(k),
                        sub_path=f".{k}",
                        record=is_recording,
                        must_find_alias=must_find_alias,
                        check=check,
                    )

            elif depth == 3:
                for k in range(arrays.Size):
                    row = arrays.entry_ref_(k)
                    for kk in range(row.Size):
                        all_variables.add_dr_var(
                            self.aliases,
                            parent_key,
                            dr_type,
                            row.entry_ref_(kk),
                            sub_path=f".{k}.{kk}",
                            record=is_recording,
                            must_find_alias=must_find_alias,
                            check=check,
                        )
            else:
                raise NotImplementedError(
                    f"Not supported yet: input or return array of Depth > 3"
                    f' (found {depth} at position or key "{path}")'
                    "  in frozen function."
                )

            if path_to_type is not None:
                path_to_type[path] = (type(arrays), False, parent_path)

        elif _dr.is_struct_v(arrays):
            for k, v in arrays.DRJIT_STRUCT.items():
                self.flatten_all(
                    getattr(arrays, k),
                    all_variables,
                    f"{path}.{k}",
                    is_parsing_inputs=is_parsing_inputs,
                    is_recording=is_recording,
                    must_find_alias=must_find_alias,
                    check=check,
                    path_to_type=path_to_type,
                    parent_path=path,
                )

            if path_to_type is not None:
                path_to_type[path] = (type(arrays), True, parent_path)

        elif isinstance(arrays, (tuple, list)):
            for i, v in enumerate(arrays):
                self.flatten_all(
                    v,
                    all_variables,
                    f"{path}.{i}",
                    is_parsing_inputs=is_parsing_inputs,
                    is_recording=is_recording,
                    must_find_alias=must_find_alias,
                    check=check,
                    path_to_type=path_to_type,
                    parent_path=path,
                )

            if path_to_type is not None:
                path_to_type[path] = (type(arrays), True, parent_path)

        elif isinstance(arrays, dict):
            for k, v in arrays.items():
                self.flatten_all(
                    v,
                    all_variables,
                    f"{path}.{k}",
                    is_parsing_inputs=is_parsing_inputs,
                    is_recording=is_recording,
                    must_find_alias=must_find_alias,
                    check=check,
                    path_to_type=path_to_type,
                    parent_path=path,
                )

            if path_to_type is not None:
                path_to_type[path] = (type(arrays), True, parent_path)

        elif dataclasses.is_dataclass(arrays):
            for field in dataclasses.fields(arrays):
                v = getattr(arrays, field.name)
                self.flatten_all(
                    v,
                    all_variables,
                    f"{path}.{field.name}",
                    is_parsing_inputs=is_parsing_inputs,
                    is_recording=is_recording,
                    must_find_alias=must_find_alias,
                    check=check,
                    path_to_type=path_to_type,
                    parent_path=path,
                )

            if path_to_type is not None:
                path_to_type[path] = (type(arrays), True, parent_path)

        elif arrays is None or isinstance(arrays, ALLOWED_PYTHON_TYPES) or callable(arrays):
            if is_recording:
                self.path_to_key[path] = all_variables.add_python_value(path, arrays)
            else:
                # The kernels were recorded for a certain set of Python values as
                # function inputs. Those values were saved during recording so that
                # we can check now that they match.
                # We do not save the new value here, because it could be confusing.
                # Note that this means that "innocent" Python values that are simply
                # passed as arguments and e.g. returned later will not be supported.
                key = ArrayKey(
                    kernel_hash=None,
                    kind=ValueKind.PYTHON_VALUE,
                    path=path,
                    sub_path=None,
                )
                if key not in all_variables.python_values:
                    # TODO: more general checking of signatures & type matches
                    raise ValueError(
                        "Error when parsing the frozen function inputs: during the first launch,"
                        f" the value at {path=} was not a Python value, however it was passed as"
                        " a Python value here, which is not supported."
                    )

                if check:
                    ref = all_variables[key]
                    if type(arrays) != type(ref) or arrays != ref:
                        raise ValueError(
                            "Error preparing kernel inputs from the function arguments."
                            " When generating the kernel during the function's first run,"
                            f' parameter "{path}" had literal value {ref} (type: {type(ref)}),'
                            f" which was baked into the kernel."
                            f" However, in this function call, a different value ({arrays},"
                            f" type {type(arrays)}) was given to this argument, which will lead"
                            " to incorrect results."
                        )
            if path_to_type is not None:
                path_to_type[path] = (type(arrays), False, parent_path)

        elif is_parsing_inputs and path.endswith(".self"):
            # TODO: remove this special carve-out, there can definitely be a lot
            #       of state kept in `self`.
            pass
        else:
            raise NotImplementedError(
                f'Not supported yet: input or return type {type(arrays)} at path "{path}"'
                " in frozen function (expected only DrJit JIT-style arrays)."
                f" Simple Python types are also allowed in the inputs: {ALLOWED_PYTHON_TYPES}"
            )


class FunctionOutputsMapping(FunctionInputsMapping):

    def __init__(self):
        super().__init__()

        # Records the types that are used when nesting types
        # To do so, paths are mapped to:
        # - their corresponding types
        # - a boolean specifying whether they are nested (used for the DFS)
        # - a parent_path to check which children in the flattened array to select when performing the DFS
        self.path_to_type: Dict[str, Tuple(Type, bool, str)] = {}

    def flatten_all_outputs(
        self,
        arrays,
        all_variables: FlatVariables,
        is_recording: bool,
        check: bool = False,
    ):
        assert is_recording

        # At recording time, when going through the function's outputs,
        # we must find each array that's part of the outputs somewhere
        # in `all_variables` (i.e. the array must have been passed as
        # input or produced as output by one of the kernel, or created
        # as a pre-allocated output buffer).
        super().flatten_all(
            arrays,
            all_variables,
            path=FN_OUTPUT_PREFIX,
            is_parsing_inputs=False,
            is_recording=is_recording,
            must_find_alias=True,
            check=check,
            path_to_type=self.path_to_type if is_recording else None,
        )

    def record_inputs_grads(self, input_keys: list[ArrayKey], kwargs: Dict[str, Any], all_variables: FlatVariables):
        """
        Evaluating the function may have created gradients in the inputs, and / or
        changed the `grad_enabled()` status. We record that effect here.

        Note that even though this function could have been the responsibility of the
        *input* function mapping (`FunctionInputsMapping`), we make use of `self.aliases`
        to map gradients to recorded variables at the end of execution. If we added these
        kinds of aliases to the *input* function mapping object, it would be checked
        and expected to alias *right from the start* of subsequent executions, which
        is not what we want.
        """
        for key in input_keys:
            if (key.kind != ValueKind.COMPOSITE_ARRAY) or (key not in all_variables.grad_enabled_before):
                # All differentiable DrJit arrays we aim to handle here are composite arrays.
                continue

            new_value = get_array_ref_for_key(kwargs, key, path_prefix=FN_INPUT_PREFIX)
            is_enabled = _dr.grad_enabled(new_value)
            all_variables.grad_enabled_after[key] = is_enabled

            if is_enabled:
                sub_path = (key.sub_path or "") + ".grad"
                # TODO: old grad could have been a literal, too (does that change anything?)
                old_grad_key = key._replace(kind=ValueKind.VARIABLE, sub_path=sub_path)

                new_grad = new_value.grad_()
                if new_grad.index in all_variables.var_index_to_key:
                    # This buffer was already part of the inputs, an explicit kernel
                    # output, or an `unclaimed_array` that was written to via a pointer.
                    new_grad_key = all_variables.var_index_to_key[new_grad.index]
                elif is_literal(new_grad):
                    # Just a default zero-valued gradient, let's add it to the literals.
                    # In some cases the right zero-valued literal will be part of `all_variables`
                    # from the very start, but sometimes a new zero-valued literal is
                    # created for some reason.
                    lit_value = new_grad[0]
                    width_hint = "width1" if _dr.width(new_grad) == 1 else "match_width"
                    lit_key = ArrayKey(
                        kernel_hash=None,
                        kind=ValueKind.VARIABLE,
                        path=f"{GRAD_LITERALS_PREFIX}.v{lit_value}.{width_hint}",
                        sub_path=None,
                    )
                    # The grad literal probably has width equal to the primal value's width.
                    # We save it as width-1, so that we can easily resize it to the appropriate
                    # width in subsequent launches.
                    if lit_key in all_variables:
                        # We may have done this already for another variable, just reuse the same literal
                        new_grad_key = lit_key
                    else:
                        new_grad_key = all_variables.add_dr_var(
                            self.aliases, lit_key, type(new_grad), type(new_grad)(lit_value), record=True
                        )
                        assert lit_key == new_grad_key

                else:
                    raise KeyError(
                        f"Function input {key=} has a gradient with index {new_grad.index}, but we"
                        f" could not find that variable index anywhere in the recorded inputs,"
                        f" outputs, or pre-allocated arrays."
                    )

                if new_grad_key != old_grad_key:
                    self.aliases[old_grad_key] = new_grad_key
                # TODO: interplay with renumbered variables?
                # TODO: new input grads that are literals?

    def set_input_grads(self, kwargs: Dict[str, Any], all_variables: FlatVariables):
        """
        Evaluating the function may have created gradients in the inputs, and / or
        changed the `grad_enabled()` status. We reproduce that effect here.
        """
        for key, is_enabled in all_variables.grad_enabled_after.items():
            if not key.path.startswith(FN_INPUT_PREFIX):
                continue
            if not is_enabled and not all_variables.grad_enabled_before[key]:
                continue

            entry_ref = get_array_ref_for_key(kwargs, key, path_prefix=FN_INPUT_PREFIX)
            # Even if we don't have to output gradients, we still want to match
            # the `grad_enabled` state.
            _dr.set_grad_enabled(entry_ref, is_enabled)
            if not is_enabled:
                continue

            # The key stored in `self.grad_enabled` doesn't include the `.grad` subpath.
            grad_key = key._replace(kind=ValueKind.VARIABLE, sub_path=(key.sub_path or "") + ".grad")
            # Resolve aliases, since the grad array may not have initially appeared as
            # that particular array's gradient. If we didn't register an alias for that
            # grad key, then:
            # - There were no gradient changes for that variable (nothing to do).
            # - Or, new gradients were scattered directly to the existing grad buffer
            #   that came with that variable from the start (nothing to do).
            if grad_key in self.aliases:
                grad_key = self.aliases[grad_key]

                new_grad = all_variables.flat_variables[grad_key]
                if grad_key in all_variables.literals:
                    # When grads are enabled, zero-valued literals are sized to the width of
                    # the primal value. But it's still possible that the user explicitly set
                    # some gradient value as a width-1 literal. So we try to support all possibilities.
                    width_mismatch = _dr.width(new_grad) not in (1, _dr.width(entry_ref))
                    if grad_key.path.endswith("match_width") or width_mismatch:
                        _dr.resize(new_grad, _dr.width(entry_ref))
                        assert is_literal(new_grad)  # TODO: remove this

                _dr.set_grad(entry_ref, new_grad)
            else:
                pass

    def to_function_outputs(self, all_variables: FlatVariables):
        return self._reconstruct_nested(all_variables, FN_OUTPUT_PREFIX)

    def _reconstruct_nested(self, all_variables: FlatVariables, current_path: str):
        """
        Auxiliary function to handle composite return types.
        """
        current_type, is_nested, _ = self.path_to_type[current_path]
        if current_type in (tuple, list):
            # Note: we assume entries were added to `self.path_to_key` in
            # the correct list order.
            result = current_type(
                (
                    self._reconstruct_nested(all_variables, children_path)
                    if self.path_to_type[children_path][1]
                    else self._reconstruct_single_array(all_variables, self.path_to_key[children_path])
                )
                for children_path, _ in self.path_to_type.items()
                if self.path_to_type[children_path][2] == current_path
            )

        elif current_type is dict:
            result = {}
            for children_path, _ in self.path_to_type.items():
                _, is_nested, parent_path = self.path_to_type[children_path]
                if parent_path == current_path:
                    result[children_path[len(current_path) + 1 :]] = (
                        self._reconstruct_nested(all_variables, children_path)
                        if is_nested
                        else self._reconstruct_single_array(all_variables, self.path_to_key[children_path])
                    )

        elif is_nested:
            # DRJIT_STRUCT or dataclass
            result = {}
            for children_path, _ in self.path_to_type.items():
                _, is_nested, parent_path = self.path_to_type[children_path]
                if parent_path == current_path:
                    result[children_path[len(current_path) + 1 :]] = (
                        self._reconstruct_nested(all_variables, children_path)
                        if is_nested
                        else self._reconstruct_single_array(all_variables, self.path_to_key[children_path])
                    )

            result = current_type(**result)

        elif current_type is type(None):
            result = None

        else:
            assert len(self.path_to_key) == 1
            result = self._reconstruct_single_array(all_variables, self.path_to_key[current_path])

        return result

    def _reconstruct_single_array(self, all_variables: FlatVariables, parent_key: ArrayKey):
        assert parent_key.sub_path is None
        if parent_key.kind == ValueKind.PYTHON_VALUE:
            # Plain Python value, let's assume it must be constant between recording
            # and replay time and just return the one that we saved.
            return all_variables[parent_key]

        # Under the key of the function outputs, the DrJit type was recorded.
        # We get it now, because we can't expect DrJit types to have been
        # recorded from the raw kernel outputs.
        parent_type = all_variables.dr_types[parent_key]
        is_diff = is_really_diff_v(parent_type)
        depth = parent_type.Depth
        key = parent_key._replace(kind=ValueKind.VARIABLE)

        # At recording time, we flattened the function outputs and added them
        # to `all_variables`. Assuming that none of these function outputs
        # appeared out of nowhere, all of the DrJit-typed values were already
        # present in `all_variables`. We use `self.aliases` to find the key
        # under which the value we are looking for was actuall stored.
        if depth == 1:
            result = all_variables.get_dr_var(key, is_diff, aliases=self.aliases)
        elif depth == 2:
            return parent_type(
                *(
                    all_variables.get_dr_var(key._replace(sub_path=f".{k}"), is_diff, aliases=self.aliases)
                    for k in range(parent_type.Size)
                )
            )
        else:
            assert depth == 3
            return parent_type(
                *(
                    parent_type.Value(
                        *(
                            all_variables.get_dr_var(
                                key._replace(sub_path=f".{k}.{kk}"),
                                is_diff,
                                aliases=self.aliases,
                            )
                            for kk in range(parent_type.Value.Size)
                        )
                    )
                    for k in range(parent_type.Size)
                )
            )

        return result


def check_closure(fn_name: str, fn: Callable):
    closure_seen = set()
    # Keep any DrJit array found in the closure and a reference to their
    # parent, for pointer analysis.
    closure_arrays: Dict[int, Tuple[_dr.ArrayBase, object, str]] = {}

    def check_recursive(fn_name, fn):
        if fn in closure_seen:
            return
        closure_seen.add(fn)
        closure_vars = inspect.getclosurevars(fn)
        annotations_mod = inspect.get_annotations(inspect.getmodule(fn))

        for source_name, source in (
            ("global", closure_vars.globals),
            ("nonlocal", closure_vars.nonlocals),
        ):
            annotations_source = deepcopy(annotations_mod)
            try:
                annotations_source.update(inspect.get_annotations(source))
            except TypeError:
                pass

            for name, value in source.items():
                fail = False

                annotations = deepcopy(annotations_source)
                try:
                    annotations.update(inspect.get_annotations(value))
                except TypeError:
                    pass

                if inspect.ismodule(value) or inspect.isclass(value):
                    # Allowed
                    pass

                elif callable(value):
                    try:
                        check_recursive(name, value)
                    except TypeError:
                        if inspect.isbuiltin(value):
                            # Allow C extensions (let's hope there is no
                            # hidden state on the C side, we can't control that).
                            pass
                        elif hasattr(value, "__call__"):
                            # This allows for some state in callable objects,
                            # which is not ideal. But we cannot be too stringent
                            # either.
                            try:
                                check_recursive(name, value.__call__)
                            except TypeError:
                                # If we can not recursively obtain closure objects
                                # we still should not fail, because there are
                                # common cases where this happens (e.g. we can
                                # not peek into external binaries) that could still
                                # be fine. Basically, it is up to the user to be
                                # sane with their state at this point.
                                pass
                        else:
                            fail = True

                elif name in ("__name__", "__module__"):
                    # Allowed
                    pass

                elif name in annotations:
                    annotated_type = annotations[name]
                    fail = not (
                        annotated_type == "Final"
                        or annotated_type == "typing.Final"
                        or annotated_type.startswith("Final[")
                        or annotated_type.startswith("typing.Final[")
                        or annotated_type == Final
                        or typing.get_origin(annotated_type) == Final
                    )

                elif hasattr(value, "__dataclass_params__") and getattr(value.__dataclass_params__, "frozen", False):
                    # Allowed: frozen dataclass instances
                    pass

                else:
                    fail = True

                if fail:
                    raise RuntimeError(
                        f'Not supported: closure (captured) {source_name} variable "{name}"'
                        f" (type {type(value)}) used in frozen function (function name: {fn_name})."
                        f" If you would like to explicitly allow this variable (and it truly"
                        f" remains constant, please annotate it with a subtype of `typing.Final`, e.g.:\n"
                        f"    {name}: Final[{type(value).__name__}] = ...\n"
                        f"The annotations found were: {annotations}"
                    )

    check_recursive(fn_name, fn)
    return closure_arrays


class FrozenKernel:
    def __init__(self, launch_i: int, history_entry: dict):
        self.history_entry: dict = history_entry
        self.kernel_type: _dr.KernelType = self.history_entry["type"]

        if self.kernel_type == _dr.KernelType.JIT:
            self.kernel_ir: str = self.history_entry["ir"].read()
            self.hash: KernelHash = (
                launch_i,
                self.history_entry["hash_low"],
                self.history_entry["hash_high"],
            )
        elif self.kernel_type == _dr.KernelType.Reduce:
            self.reduction_op: _dr.ReduceOp = self.history_entry["reduction_op"]
            # We still need some kind of hash as part of `ArrayKey`s.
            self.hash = (launch_i, None, None)
        else:
            # TODO: try and support those as well
            raise NotImplementedError(
                "Not supported yet: special kernels such as `dr.prefix_sum()`, `dr.block_reduce()`, etc."
                f" Kernel history entry {launch_i}: {history_entry}"
            )

        self.backend: _dr.JitBackend = self.history_entry["backend"]

        # print('\n------------')
        # print(self.kernel_ir)
        # print('\n------------\n')

        # Support for an edge case where the only actual inputs left
        # are pointers to variables. We have to infer the launch size
        # based on the pointers' source variable. However, the launch
        # size is not necessary equal to the width of the source.
        # For example:
        #     idx = dr.arange(UInt32, 0, dr.width(v), 5)
        #     return dr.gather(Float, v, idx)
        # In that case, the launch size should be dr.width(v) // 5.
        # We record the original launch size and inputs size, and
        # use a heuristic to try and guess the updated launch
        # size when given new inputs.
        self.original_launch_size: int = self.history_entry["size"]
        self.original_input_size: int = None
        self.original_launch_size_ratio: Tuple[bool, int, bool] = None

        self.inputs_mapping: KernelInputsMapping = None

        # Actual component type for each inner-most JIT array of the outputs,
        # listed in kernel output order. Subsequent launches require us to know
        # the raw DrJit variable type of outputs, so that they can be allocated.
        self.output_raw_types: List[_dr.VarType] = [tp for _, tp, _ in self.history_entry["output_variables"]]

        # Returns a mapping from actual kernel parameter slot (which can contain either
        # an input or output pointer) to a pair indicating:
        # - bool: whether the entry is an input or an output
        # - int: index in the inputs passed to `dr.launch()` (resp. the outputs, in the same
        #        order as the `return_types` argument passed to `dr.launch()`).
        self.kernel_slots_mapping: List[Tuple[bool, int]] = None

    def n_inputs(self):
        """Number of variables actually seen by the kernel."""
        return len(self.inputs_mapping.key_to_kernel_idx)

    def n_outputs(self):
        return len(self.output_raw_types)

    def record_inputs_mapping(self, all_variables: FlatVariables, unclaimed_arrays: Dict[int, _dr.ArrayBase]):
        # Build a mapping from parameter name to positions in the list of low-level
        # kernel inputs. Note that a single function argument can map to multiple kernel
        # inputs, e.g. if it is an Array3f.
        self.inputs_mapping = KernelInputsMapping()
        self.inputs_mapping.record_kernel(
            all_variables,
            self,
            is_parsing_inputs=True,
            unclaimed_arrays=unclaimed_arrays,
        )
        if self.history_entry["input_count"] != self.n_inputs():
            raise RuntimeError(
                f'Unexpected: the kernel input count ({self.history_entry["input_count"]})'
                f" does not match the number of DrJit variables passed to the"
                f" frozen function ({self.n_inputs()})."
            )

        self.original_input_size = self.inputs_mapping.max_width
        self.original_launch_size_ratio = size_ratio(self.original_launch_size, self.original_input_size)

    def record_outputs(self, all_variables: FlatVariables):
        # Record output variables into `all_variables` so that they can be
        # used in subsequent kernel launches' inputs as well as the overall
        # Python function outputs.
        # Note that we don't have access to the actual array variable, just
        # its index and type, but that's okay.
        output_var_idx_to_kernel_idx = {}
        for i, (var_idx, tp, size) in enumerate(self.history_entry["output_variables"]):
            all_variables.add_kernel_output(self, var_idx, tp, size, path=f"{KERNEL_OUTPUT_PREFIX}.{i}")
            output_var_idx_to_kernel_idx[var_idx] = i

        # Build the "final" kernels slots mapping
        self.kernel_slots_mapping = []
        for var_idx, _, _ in self.history_entry["all_variables"]:
            key = all_variables.var_index_to_key[var_idx]
            if var_idx in output_var_idx_to_kernel_idx:
                assert key.kind in (ValueKind.VARIABLE, ValueKind.POINTER), f"Unexpected kind for kernel output: {key=}"
                is_input = False
                kernel_idx = output_var_idx_to_kernel_idx[var_idx]
            else:
                assert key.kind in (
                    ValueKind.VARIABLE,
                    ValueKind.POINTER,
                    ValueKind.PREALLOCATED_OUTPUT,
                ), f"Unexpected kind for kernel input: {key=}"
                is_input = True
                kernel_idx = self.inputs_mapping.key_to_kernel_idx[key]
            assert kernel_idx is not None

            self.kernel_slots_mapping.append((is_input, kernel_idx))

    def add_outputs(self, all_variables: FlatVariables, kernel_outputs: List[_dr.ArrayBase]):
        """
        At launch time, add the outputs received from this kernel into `all_variables`.
        """
        for i, arr in enumerate(kernel_outputs):
            all_variables.add_kernel_output_real(self, arr, path=f"{KERNEL_OUTPUT_PREFIX}.{i}")

        # We may still hold write-enabled pointers to some of the input arrays,
        # which DrJit will interpret as pending side-effects, leading the target
        # arrays to be considered dirty. We explicitly release them here.
        all_variables.remove_pointer_variables(self)

    def prepare_kernel_inputs(self, all_variables: FlatVariables) -> FlatVariables:
        # Go over the inputs that are already available to determine the
        # launch size. We need to do this first because pre-allocated output
        # buffers probably have a size that depends on the launch size.
        max_width = 0
        for key in self.inputs_mapping.kernel_idx_to_key:
            if key in all_variables:
                max_width = max(max_width, _dr.width(all_variables[key]))

        # Create the pre-allocated output buffers for this kernel, if any
        all_variables.preallocate_buffers(self, max_width)
        all_variables.create_pointer_variables(self)

        kernel_inputs = []
        pointer_inputs = []
        for key in self.inputs_mapping.kernel_idx_to_key:
            value = all_variables[key]
            _dr.schedule(value)
            kernel_inputs.append(value)

            if key in all_variables.pointers:
                source_key, _, is_write = all_variables.pointers[key]
                pointer_inputs.append((is_write, all_variables[source_key]))

        if len(kernel_inputs) != self.n_inputs():
            raise ValueError(
                "Error preparing kernel inputs from the function arguments."
                " When generating the kernel during the function's first run,"
                f" there were {self.n_inputs()} actual kernel inputs (excluding"
                f" literals). However, in this function call, we found {len(kernel_inputs)}"
                " actual input arrays. Ensure that the same types and kind (literal or"
                " actual variable) are passed as during the first call to this function."
            )

        launch_size = self._guess_launch_size(max_width, pointer_inputs, kernel_inputs)
        return kernel_inputs, launch_size

    def launch(self, kernel_inputs: List[_dr.ArrayBase], launch_size: int):
        if self.kernel_type == _dr.KernelType.JIT:
            outputs_flat = _dr.launch(
                self.hash[1:],
                self.kernel_ir,
                self.output_raw_types,
                kernel_inputs,
                self.kernel_slots_mapping,
                backend=self.backend,
                size=launch_size,
            )
            assert len(outputs_flat) == len(self.output_raw_types)
        elif self.kernel_type == _dr.KernelType.Reduce:
            assert (
                len(kernel_inputs) == 2
            ), "Reduction kernel launch expected two inputs: the input and output pointers."

            _dr.jit_reduce(
                self.backend,
                kernel_inputs[0].Type,
                self.reduction_op,
                kernel_inputs[0].data_(),
                launch_size,
                kernel_inputs[1].pointer_value(),
            )
            # Output: none. The variable that the write-enabled pointer is pointing to
            # will be set as the function output automatically by our logic if needed.
            outputs_flat = []
        else:
            raise NotImplementedError(f"Launching a special kernel with type {self.kernel_type}.")

        # We have to make sure that all relevant outputs are part of the schedule
        _dr.eval(outputs_flat)
        return outputs_flat

    def _guess_launch_size(self, max_input_size, pointer_inputs, kernel_inputs) -> int:
        launch_size = max_input_size
        if len(kernel_inputs) == 0:
            # Since there are no non-literal kernel inputs, we won't be
            # able to infer the launch size from them. Use the very first
            # call's launch size instead.
            launch_size = self.original_launch_size

        # Edge case: the only input ends up being size-1 pointers.
        if len(pointer_inputs) == len(kernel_inputs):
            # We don't want the launch size to be inferred to 1,
            # we need to use the source variable's width instead.
            # We use a heuristic to guess the launch size based on
            # the first launch parameters, since they may not be equal.
            is_multiple, factor, valid = self.original_launch_size_ratio

            if not valid:
                raise ValueError(
                    "FrozenKernel: given that the kernel takes no array input,"
                    " we need to resort to a heuristic to guess the launch size."
                    " However, we could not guess an appropriate launch size based"
                    f" on the original input size ({self.original_input_size})"
                    f" and launch size ({self.original_launch_size})."
                )

            for _, source_var in pointer_inputs:
                candidate = apply_factor_or_ratio(_dr.width(source_var), is_multiple, factor)
                launch_size = max(launch_size, candidate)

        return launch_size


class FrozenFunction:
    """
    A `FrozenFunction` is made of one or more `FrozenKernel`s.
    """

    def __init__(self, fn, state: Callable = None, enabled: bool = True, check: bool = False):
        # Index reuse for trivial values such as Float32(0) (LVN) is only possible
        # as long as the aliased variable is alive. For AD arrays, it is quite important
        # that the default zero-valued gradient array returned by `dr.grad()` does not
        # get LVNed sometimes, and sometimes get unique variable indices.
        # Otherwise, it will be detected as a signature change for the function and
        # the frozen function would be invalid.
        # We therefore hold zero-valued Float literal arrays here to ensure that
        # LVN remains possible throughout to support this important special case.
        self.lvn_literals = {
            "a_f16": _dr.cuda.Float16(0),
            "a_f32": _dr.cuda.Float32(0),
            "a_f64": _dr.cuda.Float64(0),
        }

        self.fn = fn
        self.enabled = enabled
        # Whether to check at each call that literals and Python inputs
        # have not changed value (slower).
        self.check = check
        # A user-provided lambda which, when given the current call arguments as kwargs,
        # returns a list of latent / hidden state arrays. This is useful e.g. for params
        # that are held on the C++ side, and / or not passed explicitly as arguments.
        self.user_state: Callable = state or (lambda **_: tuple())

        # Python-level function signature
        self.signature: inspect.Signature = inspect.signature(fn)

        # Disallow *args and **kwargs in the frozen function
        for param in self.signature.parameters.values():
            if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                raise ValueError(
                    "Not supported: variable positional or keyword arguments in frozen function signature."
                )

        # Disallow closure variables in the frozen function: we want
        # it to be as "pure" as possible to avoid bad surprises.
        # The references to the collected arrays found in the closure will
        # be cleared after the first call to `fn`.
        # TODO: actually do something with `closure_arrays`.
        self.closure_arrays = check_closure(self.fn.__name__, self.fn)

        # Mapping from kernel hash to `FrozenKernel`.
        # The kernels will be recorded to this dict, and subsequently launched,
        # in the order they are listed in the DrJit kernel history.
        self.kernels: Dict[KernelHash, FrozenKernel] = None

        # An instance of `Flatvariables` is filled during recording. In subsequent
        # launches, we need to refer to the structure that was recorded, as well
        # as e.g. some literals as Python values. These are held in this instance.
        self.all_variables_template: FlatVariables = None

        # Object holding the necessary information to record the function's input,
        # and check that they match in subsequent launches.
        self.fn_inputs_mapping: FunctionInputsMapping = FunctionInputsMapping()

        # Object holding the necessary information to map the collected outputs into the
        # final function outputs.
        self.outputs_mapping: FunctionOutputsMapping = FunctionOutputsMapping()

    def __call__(self, *args, **kwargs):
        if not self.enabled or not _dr.flag(_dr.JitFlag.KernelFreezing):
            return self.fn(*args, **kwargs)

        self.args_to_kwargs(args, kwargs)
        del args

        # Implementation detail: the LVN values are looked-up per "scope" by DrJit.
        # Scopes are created during loop recording, per-thread, etc. It's easier to
        # create the literals we want to be available for LVN right now, to make sure
        # in the JIT scope that's currently used.
        self.lvn_literals["b_f16"] = _dr.cuda.Float16(0)
        self.lvn_literals["b_f32"] = _dr.cuda.Float32(0)
        self.lvn_literals["b_f64"] = _dr.cuda.Float64(0)

        # TODO: support multiple signatures (and warn if there are too many)
        if self.kernels is None:
            # --- First launch:
            # Record and analyze the function.
            output = self.trace(kwargs)

            # Done with tracing, we don't want to keep these variables
            # alive longer than they need to.
            del self.closure_arrays
        else:
            # --- Subsequent launches:
            # Launch the kernels without running the Python function.

            # Add user-provided state variables as hidden inputs to the function
            self.add_user_state(kwargs)
            # Copy over the part of `all_variables` that was saved from recording time,
            # e.g. literals. We will then add inputs, intermediate kernel outputs,
            # etc to this collection.
            all_variables = self.all_variables_template.copy_structure_shallow()

            with _dr.resume_grad():
                self.fn_inputs_mapping.flatten_all(
                    kwargs,
                    all_variables,
                    path=FN_INPUT_PREFIX,
                    is_parsing_inputs=True,
                    is_recording=False,
                    check=self.check,
                )

                for kernel in self.kernels.values():
                    kernel_inputs, launch_size = kernel.prepare_kernel_inputs(all_variables)
                    # Evaluate the inputs that were scheduled when collecting them
                    _dr.eval()

                    outputs_flat = kernel.launch(kernel_inputs, launch_size)
                    kernel.add_outputs(all_variables, outputs_flat)

                # Assemble final outputs, matching the original function's output type
                output = self.outputs_mapping.to_function_outputs(all_variables)
                all_variables.renumber_inputs(kwargs)
                self.outputs_mapping.set_input_grads(kwargs, all_variables)

                # Evaluate any last pending side-effects
                _dr.eval()

        return output

    def trace(self, kwargs):
        # --- Main function inputs
        # Add user-provided state variables as hidden inputs to the function
        kwargs_with_state = dict(kwargs)
        self.add_user_state(kwargs_with_state)
        with _dr.resume_grad():
            _dr.eval(kwargs_with_state, _dr.grad(kwargs_with_state))

            # We record input variables now, because they may end up being re-numbered
            # or modified inside of the function (e.g. by a `scatter`).
            all_variables = FlatVariables()
            self.fn_inputs_mapping.flatten_all(
                kwargs_with_state,
                all_variables,
                path=FN_INPUT_PREFIX,
                is_parsing_inputs=True,
                is_recording=True,
                check=self.check,
            )
            # Any variable with a higher index is likely to have been created
            # inside of the frozen function we're about to execute.
            # (Let's hope we don't hit the LVN cache with this variable...)
            all_variables.index_lower_bound = _dr.cuda.Int64(id(self)).index

        # TODO: avoid messing up the user's usage of KernelHistory
        _dr.kernel_history_clear()

        with _dr.scoped_set_flag(_dr.JitFlag.KernelHistory, True):
            with _dr.scoped_set_flag(_dr.JitFlag.KernelFreezing, False):
                with _dr.scoped_set_flag(_dr.JitFlag.ValueNumbering, False):
                    fn_results = self.fn(**kwargs)
                # Note that re-evaluate the inputs, in case a side effect was scheduled
                # on one of them as part of the function.
                with _dr.resume_grad():
                    _dr.eval(
                        kwargs,
                        kwargs_with_state,
                        fn_results,
                        _dr.grad(kwargs),
                        _dr.grad(kwargs_with_state),
                        _dr.grad(fn_results),
                    )
                history = _dr.kernel_history()

        if len(history) == 0:
            raise NotImplementedError(
                "dr.kernel(): running the function did not result "
                "in a DrJit kernel being generated. Cannot freeze "
                "the function."
            )

        with _dr.resume_grad():
            # --- Function outputs that are not part of the kernel outputs
            # Some of these may be new arrays that were allocated in order to write to them.
            # In that case, we will probably find a write-enabled pointer to this array in one
            # of the kernel's inputs.
            # If we find such a pointer, we will have to pre-allocate an array of this type
            # at each launch so that it can be written to.
            # TODO: case of array allocated for one kernel, and written to in a later one? Should be okay?
            # TODO: do we need to do this progressively, kernel-per-kernel or all upfront is okay?
            all_outputs_tmp = FlatVariables()
            FunctionInputsMapping().flatten_all(
                kwargs_with_state,
                all_outputs_tmp,
                path="__dummy_input",
                is_parsing_inputs=True,
                is_recording=True,
                check=False,
            )
            FunctionInputsMapping().flatten_all(
                fn_results,
                all_outputs_tmp,
                path="__dummy_output",
                is_parsing_inputs=False,
                is_recording=True,
                check=False,
            )
            unclaimed_arrays = all_outputs_tmp.get_unclaimed_arrays(history)
            del all_outputs_tmp

            # TODO: remove this
            if False:
                for launch_i, history_entry in enumerate(history):
                    print(f"\nKernel {launch_i}:")
                    print(f"- {history_entry['pointer_variables']=}")
                    print(f"- {history_entry['input_variables']=}")
                    print(f"- {history_entry['output_variables']=}")
                print(f"{unclaimed_arrays=}:")

            # --- Per-kernel inputs and outputs
            prev_kernel = None
            self.kernels = {}
            for launch_i, history_entry in enumerate(history):
                if history_entry.get("is_frozen", False):
                    raise RuntimeError(
                        "Kernel freezing was globally disabled while recording the frozen function,"
                        " but we still found a frozen kernel launch inside the function."
                        " This is not suppposed to happen and is not supported."
                    )

                kernel = FrozenKernel(launch_i, history_entry)
                kernel.record_inputs_mapping(all_variables, unclaimed_arrays)
                kernel.record_outputs(all_variables)

                if prev_kernel and kernel.backend != prev_kernel.backend:
                    raise NotImplementedError(
                        "All recorded kernels must run on the same backend "
                        f"(found {prev_kernel.backend} != {kernel.backend})"
                    )

                # Note: ordered dict, we will replay the kernels in the order they were added here.
                assert kernel.hash not in self.kernels
                self.kernels[kernel.hash] = kernel
                prev_kernel = kernel

            # --- Python function outputs
            # Now that we have collected all inputs and outputs from all kernels,
            # we can record their mapping to the overall Python function outputs.
            self.outputs_mapping.flatten_all_outputs(fn_results, all_variables, is_recording=True, check=self.check)

            # --- Function inputs that may have been renumbered
            # We run this detection step last so that we can directly refer to the
            # pre-allocated arrays that the variable was renumbered to, if any.
            all_variables.record_renumbered_variables()

            # --- Function inputs that may have gained or lost some gradients
            self.outputs_mapping.record_inputs_grads(
                self.fn_inputs_mapping.path_to_key.values(), kwargs_with_state, all_variables
            )

        # Hold on to the structure for subsequent launches
        self.all_variables_template = all_variables.copy_structure_shallow()

        return fn_results

    def args_to_kwargs(self, args, kwargs):
        """
        Fills `kwargs` in place with `args` using the
        function signature. Unspecified default arguments
        are filled-in as well.
        """
        for i, (name, param) in enumerate(self.signature.parameters.items()):
            if i < len(args):
                # Add positional arguments
                assert name not in kwargs
                kwargs[name] = args[i]
            elif name not in kwargs:
                # Add default values for missing keyword arguments
                if param.default == param.empty:
                    raise ValueError(
                        f'Function argument "{name}" was not passed,'
                        " but it did not have a default value in the"
                        " function signature."
                    )
                assert param.default != param.empty, (name, param)
                kwargs[name] = param.default

    def add_user_state(self, kwargs):
        user_state = self.user_state(**kwargs)
        if (user_state is not None) and not isinstance(user_state, (list, tuple)):
            raise TypeError(f"The `state` lambda function must return a list or tuple, found {type(user_state)}.")
        for i, v in enumerate(user_state):
            kwargs[USER_STATE_PREFIX + str(i)] = v


# Hold weak references to frozen functions seen so far, so that we can
# clear them all at once in `dr.clear_frozen_kernels()`.
# We use weak references because we don't want to keep function objects
# alive just to be able to clear them later.
_ALL_FROZEN_FUNCTIONS = WeakSet()


def kernel(fn=None, state: Callable = None, enabled: bool = True, check: bool = False):
    """
    Function decorator.
    Traces the function only once, subsequent calls result in
    direct kernel launches, which removes the Python and tracing overhead.

    The traced function must be "pure" (as much as that is possible in Python)
    and not have any control flow that changes based on the inputs (e.g. a Python boolean).

    Inputs:
        check: at every call, check that DrJit literals and Python values in the inputs
               of the function have the same values as were recorded during the first run.
               It is recommended to enable this when debugging to ensure that the frozen
               kernel will produce correct resuls.
    """

    def decorator(fn):
        # TODO: make sure we get correct AD results
        frozen = FrozenFunction(fn, state=state, enabled=enabled, check=check)

        @functools.wraps(fn)
        def proxy(*args, **kwargs):
            # We introduce this intermediary so that `self` is not
            # overwritten (?) by our `FrozenFunction` instance.
            return frozen(*args, **kwargs)

        # TODO: cleaner way to achieve this
        def enable_freezing():
            nonlocal frozen
            frozen.enabled = True

        def disable_freezing():
            nonlocal frozen
            frozen.enabled = False

        def set_frozen(v: bool):
            nonlocal frozen
            frozen.enabled = v

        def clear():
            """
            Clears any previously recorded kernel and resets the options (enabled, etc)
            to the values passed to the decorator.
            The next time the function is called while this FrozenFunction
            is enabled, the recording will take place again.
            """
            nonlocal frozen
            frozen = FrozenFunction(fn, state=state, enabled=enabled, check=check)

        proxy.__dict__["freeze"] = enable_freezing
        proxy.__dict__["unfreeze"] = disable_freezing
        proxy.__dict__["is_frozen"] = lambda: frozen.enabled
        proxy.__dict__["set_frozen"] = set_frozen
        proxy.__dict__["frozen"] = frozen
        proxy.__dict__["clear"] = clear

        _ALL_FROZEN_FUNCTIONS.add(proxy)
        return proxy

    if fn is not None:
        return decorator(fn)
    else:
        return decorator


def clear_frozen_kernels():
    """Clears the recorded state of all existing `FrozenFunction` instances."""
    for proxy in _ALL_FROZEN_FUNCTIONS:
        proxy.clear()
