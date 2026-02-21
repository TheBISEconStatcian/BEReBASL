from typing import Literal, Optional

import torch


def assert_tensors(
        *tensors, 
        tensor_names : Optional[str],
        checks : List[]
    ):
    if len(tensors) == 0:
        return
    pass