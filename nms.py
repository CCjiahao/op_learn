from typing import Tuple
import torch
from torchvision.ops import nms


def generate_boxes(len: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    # 随机生成boxes
    boxes = torch.rand(len, 4) * 100
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
    scores = torch.rand(len)
    return boxes, scores


def my_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    # 预处理面积
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 排序
    orders = scores.argsort(descending=True)

    keep = []
    while orders.numel() > 0:
        if orders.numel() == 1:
            keep.append(orders.item())
            break
        else:
            keep.append(orders[0].item())

        lt = torch.max(boxes[orders, None, :2], boxes[orders[:1], :2])
        rb = torch.min(boxes[orders, None, 2:], boxes[orders[:1], 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0, 0] * wh[:, 0, 1]
        iou = inter / (area[orders[0]] + area[orders] - inter)
        orders = orders[(iou < iou_threshold).nonzero().squeeze()]
    return torch.LongTensor(keep)


def test_iou(boxes: torch.Tensor, scores: torch.Tensor):
    std_ids = nms(boxes, scores, 0.5)
    my_ids = my_nms(boxes, scores, 0.5)
    assert torch.allclose(std_ids, my_ids)


def main():
    boxes, scores = generate_boxes(100)
    test_iou(boxes, scores)


if __name__ == '__main__':
    main()
