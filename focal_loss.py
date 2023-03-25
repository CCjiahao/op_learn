import torch
from torch import nn
from torchvision.ops import box_iou


class FocalLoss(nn.Module):
    ''' Focal Loss '''
    def __init__(self) -> None:
        super().__init__()
        self.alpha = 0.25
        self.gamma = 2.0
        self.negative_threshold = 0.4
        self.positive_threshold = 0.5

    @staticmethod
    def get_box_regression_target(source_anchors: torch.Tensor, target_anchors: torch.Tensor) -> torch.Tensor:
        ''' get box regression target '''
        source_anchors_wh = source_anchors[:, 2:] - source_anchors[:, :2]
        source_anchors_xy = source_anchors[:, :2] + 0.5 * source_anchors_wh
        
        target_anchors_wh = target_anchors[:, 2:] - target_anchors[:, :2]
        target_anchors_xy = target_anchors[:, :2] + 0.5 * target_anchors_wh
        target_anchors_wh = torch.clamp(target_anchors_wh, min=1)

        target_anchors_xy = (target_anchors_xy - source_anchors_xy) / source_anchors_wh
        target_anchors_wh = torch.log(target_anchors_wh / source_anchors_wh)

        return torch.cat([target_anchors_xy, target_anchors_wh], dim=1)

    def forward(self, classifications: torch.Tensor, regressions: torch.Tensor, anchors: torch.Tensor, annotations: torch.Tensor):
        '''
        input:
            classifications: [B, num_anchors, num_class]
            regressions: [B, num_anchors, 4]
            anchors: [num_anchors, 4]
            annotations: [B, num_gt_boxes, 5]
        output:
        '''

        classification_losses = []
        regression_losses = []
        for i in range(classifications.shape[0]):
            classification = classifications[i]
            regression = regressions[i]
            bbox_annotation = annotations[i]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            # 计算IOU，得到每个anchor与gt box的最近匹配
            iou = box_iou(anchors, bbox_annotation[:, :4])
            iou_max, iou_argmax = torch.max(iou, dim=-1)
            assigned_annotations = bbox_annotation[iou_argmax, :]

            # ------------------------------ cls loss ---------------------------------#
            # 获取分类target，负样本target都是0，正样本target是one hot的label
            targets = -torch.ones(classification.shape)
            targets[torch.lt(iou_max, self.negative_threshold), :] = 0
            # 正样本设置为one hot
            positive_indices = torch.ge(iou_max, self.positive_threshold)
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            # 做focal loss
            ## 1. 做bce loss: - ( y*log(p(x)) + (1-y)*log(1-p(x)) )
            bce = -(targets * torch.log(classification) + (1 - targets) * torch.log(1 - classification))
            ## 2. focal weight
            alpha_factor = torch.ones(targets.shape) * self.alpha
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)      # 分类难度
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            cls_loss = focal_weight * bce
            ## 3. 去除不关注的中心样本
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))
            classification_losses.append(cls_loss.sum() / torch.clamp(positive_indices.sum().float(), min=1.0))
            # -------------------------------------------------------------------------#

            # ------------------------------ box loss ---------------------------------#
            # 获取target
            targets = self.get_box_regression_target(anchors[positive_indices, :], assigned_annotations[positive_indices, :4])
            # target正则化
            targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

            # 计算huber loss
            regression_diff = torch.abs(targets - regression[positive_indices, :])
            regression_loss = torch.where(
                torch.le(regression_diff, 1.0 / 9.0),
                0.5 * 9.0 * torch.pow(regression_diff, 2),
                regression_diff - 0.5 / 9.0
            )
            regression_losses.append(regression_loss.mean())
        return torch.stack(classification_losses).mean(), torch.stack(regression_losses).mean()


def main():
    focalLoss = FocalLoss()

    focal_loss_input = torch.load('focal_loss.pth')
    classifications = focal_loss_input['classifications']
    regressions = focal_loss_input['regressions']
    anchors = focal_loss_input['anchors'][0]
    annotations = focal_loss_input['annotations']

    classification_loss, regression_loss = focalLoss(classifications, regressions, anchors, annotations)

    assert torch.allclose(classification_loss, torch.tensor(1.161665916442871))
    assert torch.allclose(regression_loss, torch.tensor(1.0848838090896606))


if __name__ == '__main__':
    main()
