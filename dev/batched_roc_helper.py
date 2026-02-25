from matplotlib import pyplot as plt
import torch

from typing import Tuple

def generate_data(
        N: int, repeated_scores: int, proportion_y_equals_1: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device('cpu')
    rng = torch.Generator(device=device).manual_seed(1807)

    N_y_equals_1 = round(N * proportion_y_equals_1)
    N_y_equals_0  = N - N_y_equals_1
    true_prob = torch.cat([
        torch.rand(size=(N_y_equals_0,), generator=rng) * 0.4, # \in [0, 0.4)
        torch.rand(size=(N_y_equals_1,), generator=rng) * 0.4 + 0.6 # \in [0.6, 1)
    ])

    labels = torch.bernoulli(true_prob, generator=rng)

    desired_std_for_noise = 0.07 # Range of noise will be then around [-0.21,0.21]
    score_noise = torch.randn(size=(N,), generator=rng) * desired_std_for_noise
    scores = true_prob + score_noise

    repeated_scores = repeated_scores + (repeated_scores % 2)
    repeated_scores_y_1 = round(repeated_scores * proportion_y_equals_1)
    repeated_scores_y_1 = repeated_scores_y_1 + (repeated_scores_y_1 % 2)
    repeated_scores_y_0 = repeated_scores - repeated_scores_y_1

    idx_to_repeat_y0 = torch.randperm(N_y_equals_0, generator=rng)[:repeated_scores_y_0].reshape(repeated_scores_y_0//2, 2)
    idx_to_repeat_y1 = torch.randperm(N_y_equals_1, generator=rng)[:repeated_scores_y_1].reshape(repeated_scores_y_1//2, 2) + N_y_equals_0

    idx_to_repeat = torch.cat([idx_to_repeat_y0, idx_to_repeat_y1], dim=0)

    scores[idx_to_repeat[:, 0]] = scores[idx_to_repeat[:, 1]].clone()

    return scores, labels, idx_to_repeat

def generate_batched_data(
        B: int, N: int, repeated_scores: int, proportion_y_equals_1: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device('cpu')
    rng = torch.Generator(device=device).manual_seed(1807)

    N_y_equals_1 = round(N * proportion_y_equals_1)
    N_y_equals_0  = N - N_y_equals_1
    true_prob = torch.cat([
        torch.rand(size=(B, N_y_equals_0,), generator=rng) * 0.4, # \in [0, 0.4)
        torch.rand(size=(B, N_y_equals_1,), generator=rng) * 0.4 + 0.6 # \in [0.6, 1)
    ], dim=1)

    labels = torch.bernoulli(true_prob, generator=rng)

    desired_std_for_noise = 0.07 # Range of noise will be then around [-0.21,0.21]
    score_noise = torch.randn(size=(B,N), generator=rng) * desired_std_for_noise
    scores = true_prob + score_noise

    repeated_scores = repeated_scores + (repeated_scores % 2)
    repeated_scores_y_1 = round(repeated_scores * proportion_y_equals_1)
    repeated_scores_y_1 = repeated_scores_y_1 + (repeated_scores_y_1 % 2)
    repeated_scores_y_0 = repeated_scores - repeated_scores_y_1

    idx_to_repeat_y0 = torch.rand((B, N_y_equals_0), generator=rng).argsort(dim=-1)[:, :repeated_scores_y_0].reshape(B, repeated_scores_y_0//2, 2)
    idx_to_repeat_y1 = torch.rand((B, N_y_equals_1), generator=rng).argsort(dim=-1)[:, :repeated_scores_y_1].reshape(B, repeated_scores_y_1//2, 2) + N_y_equals_0

    idx_to_repeat = torch.cat([idx_to_repeat_y0, idx_to_repeat_y1], dim=1)

    scores.scatter_(dim=-1, index=idx_to_repeat[:,:,0], 
                    src=scores.gather(dim=-1, index=idx_to_repeat[:, :, 1]).clone()
    )

    return scores, labels, idx_to_repeat

def roc_curve_with_repeats_and_interp(
        fpr,
        tpr,
        sorted_repeated_mask,
        puffer: float = 0.009
) -> None:
    plt.plot(
        torch.nn.functional.pad(fpr, pad=(1,0)), 
        torch.nn.functional.pad(tpr, pad=(1,0)), 
        lw=2, color=(0.7,0.8,0.5,0.8), 
        label="Interpolated FPR/TPR = ROC"
    )
    plt.scatter(fpr[~sorted_repeated_mask], tpr[~sorted_repeated_mask], s=5, color='red',
                label="Observed FPR/TPR (unique)")
    plt.scatter(fpr[sorted_repeated_mask], tpr[sorted_repeated_mask], s=5, color='blue',
                label="Observed FPR/TPR (repeated)")
    plt.legend()

    plt.ylabel("TPR")
    plt.xlabel("FPR")

    plt.ylim(0,1 + puffer)
    plt.xlim(-puffer, 1 + puffer)

    plt.title("ROC-Curve")

    plt.show()