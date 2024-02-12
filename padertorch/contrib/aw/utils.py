from collections.abc import Mapping
import logging

import numpy as np
import torch


def nested_op_limited(
    func,
    obj,
    sequence_limit=10,
    mapping_limit=None,
    nesting_limit=None,
    handle_dataclass=False,
    show_only_first=True,
    mapping_type=Mapping,
    sequence_type=(tuple, list),
):
    """Similar to nested_op, but restricts size of collections.

    Args:
        func: function to apply to leaf objects
        obj: object of interest
        sequence_limit:
        mapping_limit: maximum number of keys, will only use the first keys returned by `keys()`.
        nesting_limit: maximum number of collections to recursively visit.
        handle_dataclass: same as in nested_op
        show_only_first: show only first element of sequence if limit is exceeded instead of all elements up to limit.
        mapping_type:
        sequence_type:

    """

    if isinstance(obj, mapping_type):
        keys = obj.keys()
        key_limit = (
            min(len(keys), mapping_limit) if mapping_limit is not None else len(keys)
        )

        if nesting_limit is None or nesting_limit > 0:
            output = {
                key: nested_op_limited(
                    func,
                    obj[key],
                    sequence_limit=sequence_limit,
                    mapping_limit=mapping_limit,
                    nesting_limit=nesting_limit - 1
                    if nesting_limit is not None
                    else None,
                    mapping_type=mapping_type,
                    sequence_type=sequence_type,
                    handle_dataclass=handle_dataclass,
                    show_only_first=show_only_first,
                )
                for key, _ in zip(keys, range(key_limit))
            }
        else:
            output = "..."
        # if keep_type:
        #     output = arg1.__class__(output)
        return output
    elif isinstance(obj, sequence_type):
        if show_only_first and len(obj) > sequence_limit:
            output = [
                nested_op_limited(
                    func,
                    obj[0],
                    sequence_limit=sequence_limit - 1
                    if sequence_limit is not None
                    else None,
                    mapping_limit=mapping_limit,
                    nesting_limit=nesting_limit,
                    mapping_type=mapping_type,
                    sequence_type=sequence_type,
                    handle_dataclass=handle_dataclass,
                    show_only_first=show_only_first,
                ),
                "...",
                f"len:{len(obj)}",
            ]
        else:
            seq_length = (
                min(len(obj), sequence_limit)
                if sequence_limit is not None
                else len(obj)
            )
            output = [
                nested_op_limited(
                    func,
                    obj[j],
                    sequence_limit=sequence_limit - 1
                    if sequence_limit is not None
                    else None,
                    mapping_limit=mapping_limit,
                    nesting_limit=nesting_limit,
                    mapping_type=mapping_type,
                    sequence_type=sequence_type,
                    handle_dataclass=handle_dataclass,
                    show_only_first=show_only_first,
                )
                for j in range(seq_length)
            ]
        # if keep_type:
        #     output = arg1.__class__(output)
        return output
    elif handle_dataclass and hasattr(obj, "__dataclass_fields__"):
        return obj.__class__(
            **{
                f_key: nested_op_limited(
                    func,
                    getattr(obj, f_key),
                    sequence_limit=sequence_limit,
                    mapping_limit=mapping_limit,
                    nesting_limit=nesting_limit - 1
                    if nesting_limit is not None
                    else None,
                    mapping_type=mapping_type,
                    sequence_type=sequence_type,
                    handle_dataclass=handle_dataclass,
                    show_only_first=show_only_first,
                )
                for f_key in obj.__dataclass_fields__
            }
        )

    return func(obj)


def str_shape_maybe(obj):
    """Print the shape if the object is a numpy array or torch."""
    # TODO: paderbox.utils.pretty?
    if torch.is_tensor(obj):
        info = obj.shape
        if len(obj.shape) == 0:  # print if scalar
            info = obj.item()
        return f"torch.Tensor({info})"
    elif isinstance(obj, np.ndarray):
        return f"np.array({obj.shape})"
    else:
        return str(obj)


def str_nested_shape_maybe(obj, **kwargs):
    return str_shape_maybe(nested_op_limited(str_shape_maybe, obj, **kwargs))