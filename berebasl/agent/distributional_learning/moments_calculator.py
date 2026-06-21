import torch
from torch import nn


class MomentsCalculator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            X_r: torch.Tensor,
            mask_r: torch.Tensor,
            X_a: torch.Tensor,
            y_a: torch.Tensor,
            mask_a: torch.Tensor
        ) -> torch.Tensor:
        *b_dims, n_a, F = X_a.shape
        n_r = mask_r.size(-1)
        mask_is_bad = y_a == 1 # [*dims, n_a]

        rbg = 3 # rejects, bads, goods

        max_n = max(n_a, n_r)
        leading_dims = b_dims + [rbg, max_n]
        

        mask_valids = mask_is_bad.new_empty(leading_dims)

        mask_valids[..., 0, :n_r] = mask_r
        mask_valids[..., 1, :n_a] = mask_is_bad & mask_a
        mask_valids[..., 2, :n_a] = ~mask_is_bad & mask_a

        count_valids = mask_valids.sum(dim=-1, keepdim=True) # [*b_dims, rbg, 1]

        X_buffer = X_r.new_empty(leading_dims + [F])
        X_buffer[..., 0, :n_r, :] = X_r
        X_buffer[..., 1:, :n_a, :] = X_a.unsqueeze(-3)


        X_buffer.masked_fill_(~mask_valids.unsqueeze(-1), 0)

        means = X_buffer.sum(dim=-2) / count_valids # [*b_dims, rbg, F]

        vcovs = X_buffer.mT @ X_buffer - count_valids.unsqueeze(-1) * (means.unsqueeze(-1) * means.unsqueeze(-2))

        vcovs /= count_valids.unsqueeze(-1) - 1 

        mask_unique_vcovs = torch.triu(vcovs.new_ones(F, F, dtype=torch.bool))

        return torch.cat([
            means, # [*b_dims, rbg, F]
            vcovs[..., mask_unique_vcovs] # [*b_dims, rbg, F*(F+1)/2]
            ], dim=-1) # [*b_dims, rbg, F*(F+3)/2]