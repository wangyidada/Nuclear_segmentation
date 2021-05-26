import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        """
        	input tesor of shape = (N, 1, H, W)
        	target tensor of shape = (N, 1, H, W)
        """
        N = target.size(0)
        smooth = 1e-2

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + 2*smooth)
        loss = 1 - loss.sum() / N
        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):
        """
        	input tesor of shape = (N, C, H, W)
        	target tensor of shape = (N, 1, H, W)
        """
        # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
        C = input.shape[1]

        target = torch.squeeze(target)
        traget_to_one_hot = nn.functional.one_hot(target.long(), num_classes=C)
        traget_to_one_hot = traget_to_one_hot.permute(0, 3, 1, 2)

        assert input.shape == traget_to_one_hot.shape, "predict & target shape do not match"

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0
        logits = nn.functional.softmax(input, dim=1)

        for i in range(1, C):
            diceLoss = dice(logits[:, i, ...], traget_to_one_hot[:, i, ...])
            if weights is not None:
                diceLoss *= weights[i - 1]
            totalLoss += diceLoss
        return totalLoss/(C-1)


class MulticlassEntropyLoss(nn.Module):
    def __init__(self):
        super(MulticlassEntropyLoss, self).__init__()

    def forward(self, input, target):
        """
        	input tesor of shape = (N, C, H, W)
        	target tensor of shape = (N, 1, H, W)
        """
        target = torch.squeeze(target)
        loss = nn.CrossEntropyLoss()(input, target.long())
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class MulitclassFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        """
        preds tesor of shape = (N, C, H, W)
        labels tensor of shape = (N, 1, H, W)
        """
        labels = torch.squeeze(labels)
        logpt = -self.ce_fn(preds, labels.long())
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
