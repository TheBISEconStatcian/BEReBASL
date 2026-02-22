from enum import Enum
import sys
from warnings import warn

from collections.abc import Iterable
from typing import Optional, Union

import torch

class Validations(Enum):
    same_device = "same_device"
    same_dtype = "same_dtype"
    same_shape = "same_shape"
    are_tensors = "are_tensors"

    def msg_completion(self):
        if self.value[:5] == "same_":
            return "did not have the same " + self.value[5:]
        if self.value=="are_tensors":
            return "were not all tensors"

def same_attribute(*tensors, attribute: str):
    try:
        expected_attr = getattr(tensors[0], attribute)
        return all([getattr(t, attribute) == expected_attr for t in tensors[1:]])
    except AttributeError:
        warn("One of the elements was not a tensor, check assumed as not passed")
        return False

def same_device(*tensors) -> bool:
    return same_attribute(*tensors, attribute="device")

def same_dtype(*tensors) -> bool:
    return same_attribute(*tensors, attribute="dtype")

def same_shape(*tensors) -> bool:
    return same_attribute(*tensors, attribute="shape")

def are_tensors(*tensors) -> bool:
    return all([isinstance(t, torch.Tensor) for t in tensors])

def assert_tensors(
        *tensors, 
        tensor_names : Optional[str],
        checks : Iterable[Union[Validations, str]],
        throw_error : bool = True
    ) -> None:
    if len(tensors) == 0:
        return
    
    if not isinstance(checks, Iterable):
        raise RuntimeError("checks was expected to be an iterable")
    
    current_module = sys.modules[__name__]
    
    for check in checks:
        validation_type = check if isinstance(check, Validations) else Validations(check)
        validation_function = getattr(current_module, validation_type.value)

        if not validation_function(*tensors):
            msg = f"{tensor_names} {validation_type.msg_completion()}"
            if throw_error:
                raise AssertionError(msg)
            
            warn(msg)