''' iou '''
import torch
from torchvision.ops import box_iou


def generate_boxes(len: int = 100) -> torch.Tensor:
    # 随机生成boxes
    boxes = torch.rand(len, 4) * 100
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
    return boxes


def my_area(boxes: torch.Tensor) -> torch.Tensor:
    # 计算面积，这里不用判断合法性
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def my_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # 计算各自面积
    area1, area2 = my_area(boxes1), my_area(boxes2)

    # 计算交集面积
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    # 计算IOU
    return inter / (area1[:, None] + area2[None, :] - inter)


def test_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    my_iou = my_box_iou(boxes1, boxes2)
    tv_iou = box_iou(boxes1, boxes2)
    assert torch.allclose(my_iou, tv_iou)


def main():
    boxes1 = generate_boxes(100)
    boxes2 = generate_boxes(200)
    test_iou(boxes1, boxes2)


if __name__ == '__main__':
    main()
