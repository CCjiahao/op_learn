import torch
from torch import nn
from torch.autograd import Function


class MyBatchNorm(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        mean = input.mean(dim=[0, 2, 3], keepdim=True)
        var = input.var(dim=[0, 2, 3], keepdim=True)
        input_ = (input - mean) / torch.sqrt(var + eps)
        ctx.save_for_backward(input_, var, weight)
        ctx.eps = eps
        return input_ * weight[:, None, None] + bias[:, None, None]

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        input_, var, weight = ctx.saved_tensors

        bias_grad = output_grad.sum(dim=[0, 2, 3])
        weight_grad = (input_ * output_grad).sum(dim=[0, 2, 3])
        input_grad = output_grad * weight[:, None, None] / torch.sqrt(var + ctx.eps)

        return input_grad, weight_grad, bias_grad, None


def my_batch_norm(input: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    mean = input.mean(dim=[0, 2, 3], keepdim=True)
    var = input.var(dim=[0, 2, 3], keepdim=True)
    return ((input - mean) / torch.sqrt(var + eps)) * beta[:, None, None] + gamma[:, None, None]


def std_bn(input, model):
    out = model(input).mean()
    out.backward()
    return out, input.grad, model.weight.grad, model.bias.grad


def my_bn(input, weight, bias, eps):
    out = MyBatchNorm.apply(input, weight, bias, eps).mean()
    out.backward()
    return out, input.grad, weight.grad, bias.grad


def main():
    input1 = torch.rand(32, 64, 48, 48, requires_grad=True)  # B, C, H, W
    input2 = input1.clone().detach().requires_grad_()

    model = nn.BatchNorm2d(64, eps=0)
    weight2 = model.weight.clone().detach().requires_grad_()
    bias2 = model.bias.clone().detach().requires_grad_()

    out1 = std_bn(input1, model)
    out2 = my_bn(input2, weight2, bias2, model.eps)

    for a, b in zip(out1, out2):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)


if __name__ == '__main__':
    main()
