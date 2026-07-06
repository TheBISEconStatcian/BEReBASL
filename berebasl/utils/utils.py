import torch

def tensor_is_not_view(ten : torch.Tensor) -> bool:
    return ten._base is None

def tensors_share_some_memory(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage() == b.untyped_storage()

def clone_if_view(ten: torch.Tensor) -> torch.Tensor:
    return ten if tensor_is_not_view(ten) else ten.clone()