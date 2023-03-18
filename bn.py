import torch
from torch import nn


def my_batch_norm(input: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    mean = input.mean(dim=[0, 2, 3], keepdim=True)
    var = input.var(dim=[0, 2, 3], keepdim=True)
    return ((input - mean) / torch.sqrt(var + eps)) * beta[:, None, None] + gamma[:, None, None]


def main():
    input = torch.rand(32, 64, 48, 48)  # B, C, H, W
    model = nn.BatchNorm2d(64)

    out1 = model(input)
    out2 = my_batch_norm(input, model.weight.data, model.bias.data, model.eps)

    assert torch.allclose(out1, out2, rtol=1e-4, atol=1e-6)

if __name__ == '__main__':
    main()
