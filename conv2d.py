import torch
from torch import nn


def main():
    input = torch.rand(4, 32, 48, 48)
    model = nn.Conv2d(32, 64, 3, padding=1)
    
    out1 = model(input)
    ########################
    ufd_input = torch.nn.functional.unfold(input, 3, padding=1)     # (B, C*k*k, h*w)
    ufd_input = ufd_input.view(4, 32, 3, 3, 48, 48)
    out2 = torch.einsum('bcklhw,ockl->bohw', [ufd_input, model.weight]) + model.bias.view(1, 64, 1, 1)
    assert torch.allclose(out1, out2, atol=1e-6)
    ###################
    pad_input = torch.nn.functional.pad(input, [1, 1, 1, 1])
    out3 = model.bias.view(1, 64, 1, 1).expand(4, 64, 48, 48)
    for k1 in range(model.weight.shape[2]):
        for k2 in range(model.weight.shape[3]):
            w = model.weight[:, :, k1, k2] 
            x = pad_input[:, :, k1:k1+48, k2:k2+48]
            out3 = out3 + torch.einsum('bchw,oc->bohw', [x, w])
    assert torch.allclose(out1, out3, atol=1e-6)


if __name__ == '__main__':
    main()
