from typing import List, Tuple
import torch
from torch import nn


class Anchors(nn.Module):
    def __init__(self):
        super().__init__()

        self._generate_base_anchors(
            4,                          # 基础尺寸
            [0.5, 1, 2],                # 宽高比
            [1, 2**(1/3), 2**(2/3)]     # 不同尺寸
        )

    def _generate_base_anchors(self, base_size: float, ratios: List[float], scales: List[float]) -> None:
        ''' ratios: 宽高比   scales: 尺寸'''
        ratios, scales = torch.tensor(ratios), torch.tensor(scales)
        w = torch.sqrt((scales**2) / ratios.unsqueeze(1))
        h = w * ratios.unsqueeze(1)
        w, h = w.view(-1), h.view(-1)
        self.base_anchors = torch.stack([-w/2, -h/2, w/2, h/2], dim=1) * base_size

    @staticmethod
    def meshgrid(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        return xx, yy

    def get_shifts(self, image_shape: List[Tuple[int]], stride: int) -> torch.Tensor:
        shift_x = (torch.arange(0.5, image_shape[1])) * stride
        shift_y = (torch.arange(0.5, image_shape[0])) * stride
        shift_x, shift_y = self.meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
        return shifts

    def forward(self, image_shapes: List[Tuple[int]], levels: List[int]) -> torch.Tensor:
        all_anchors = []
        for image_shape, level in zip(image_shapes, levels):
            stride = 2 ** level
            this_base_anchors = self.base_anchors * stride
            shifts = self.get_shifts(image_shape, stride)
            shiftd_anchors = (this_base_anchors[None, :, :] + shifts[:, None, :]).view(-1, 4)
            all_anchors.append(shiftd_anchors)
        return torch.cat(all_anchors, dim=0)


def main():
    anchors = Anchors()
    all_anchors = anchors([[76, 132], [38, 66], [19, 33], [10, 17], [5, 9]], [3, 4, 5, 6, 7])
    target_anchors = torch.load('anchors.pth')
    assert torch.allclose(all_anchors, target_anchors)


if __name__ == '__main__':
    main()
