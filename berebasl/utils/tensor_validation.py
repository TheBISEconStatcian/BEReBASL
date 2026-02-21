from enum import Enum
import sys
from warnings import warn
from typing import List, Optional, Union

import torch

class Validations(Enum):
    same_device = "same_device"
    same_dtype = "same_dtype"
    same_shape = "same_shape"

    def msg_completion(self):
        if self.value[:5] == "same_":
            return "did not have the same " + self.value[5:]

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

def assert_tensors(
        *tensors, 
        tensor_names : Optional[str],
        checks : List[Union[Validations, str]],
        throw_error : bool = True
    ) -> None:
    if len(tensors) == 0:
        return
    
    current_module = sys.modules[__name__]
    
    for check in checks:
        validation_type = check if isinstance(check, Validations) else Validations(check)
        validation_function = getattr(current_module, validation_type.value)

        if not validation_function(*tensors):
            msg = f"{tensor_names} {validation_type.msg_completion()}"
            if throw_error:
                raise AssertionError(msg)
            
            warn(msg)