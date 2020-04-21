import math
import torch
import itertools

from torchvision.ops.boxes import nms


class SSDBoxCoder:
    def __init__(self, steps, box_sizes, aspect_ratios, fm_sizes):
        self.prior_boxes = self._get_default_boxes(steps, box_sizes, aspect_ratios, fm_sizes)

    @staticmethod
    def _get_default_boxes(steps, box_sizes, aspect_ratios, fm_sizes):
        boxes = []
        for i, fm_size in enumerate(fm_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):
                cx = (w + 0.5) * steps[i]
                cy = (h + 0.5) * steps[i]

                s = box_sizes[i]
                boxes.append((cx, cy, s, s))

                s = math.sqrt(box_sizes[i] * box_sizes[i + 1])
                boxes.append((cx, cy, s, s))

                s = box_sizes[i]
                for ar in aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))
        return torch.Tensor(boxes)

    def encode(self, boxes, labels):
        '''Encode target bounding boxes and class labels.
        SSD coding rules:
          tx = (x - anchor_x) / (variance[0]*anchor_w)
          ty = (y - anchor_y) / (variance[0]*anchor_h)
          tw = log(w / anchor_w) / variance[1]
          th = log(h / anchor_h) / variance[1]
        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
        '''
        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1]
            return (i[j], j)

        device = labels.get_device()
        prior_boxes = self.prior_boxes.to(device)  # xywh
        prior_boxes = change_box_order(prior_boxes, 'xywh2xyxy')

        ious = box_iou(prior_boxes, boxes)  # [#anchors, #obj]
        # index = torch.LongTensor(len(prior_boxes)).fill_(-1).to(device)
        index = torch.full(size=torch.Size([prior_boxes.size()[0]]), fill_value=-1, dtype=torch.long, device=device)
        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i, j] < 1e-6:
                break
            index[i] = j
            masked_ious[i, :] = 0
            masked_ious[:, j] = 0

        mask = (index < 0) & (ious.max(1)[0] >= 0.5)
        if mask.any():
            index[mask] = ious[mask.nonzero().squeeze(dim=1)].max(1)[1]

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        boxes = change_box_order(boxes, 'xyxy2xywh')
        prior_boxes = change_box_order(prior_boxes, 'xyxy2xywh')

        variances = (0.1, 0.2)
        loc_xy = (boxes[:,:2]-prior_boxes[:,:2]) / prior_boxes[:,2:] / variances[0]
        loc_wh = torch.log(boxes[:,2:]/prior_boxes[:,2:]) / variances[1]
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        # cls_targets = 1 + labels[index.clamp(min=0)]  # TODO: why +1 ???
        cls_targets = labels[index.clamp(min=0)]
        cls_targets[index<0] = 0
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.05, nms_thresh=0.45):
        """Decode predicted loc/cls back to real box locations and class labels.
        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.
        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        """
        device = cls_preds.get_device() if cls_preds.get_device() >= 0 else torch.device('cpu')
        prior_boxes = self.prior_boxes.to(device)
        variances = (0.1, 0.2)
        xy = loc_preds[:, :2] * variances[0] * prior_boxes[:, 2:] + prior_boxes[:, :2]
        wh = torch.exp(loc_preds[:, 2:] * variances[1]) * prior_boxes[:, 2:]
        box_preds = torch.cat([xy - wh / 2, xy + wh / 2], 1)

        boxes = []
        labels = []
        scores = []
        # num_classes = cls_preds.size(1)
        # for i in range(1, num_classes):
        #     score = cls_preds[:, i]
        for i, cls_pred in enumerate(cls_preds.split(1, dim=1)[1:]):
            score = cls_pred.squeeze(dim=1)
            mask = (score > score_thresh).nonzero().squeeze(dim=1)
            if mask.sum() == torch.tensor(data=0, device=device):
                continue
            box = box_preds[mask]
            score = score[mask]

            # keep = box_nms(box, score, nms_thresh)
            keep = nms(box, score, nms_thresh)
            boxes.append(box[keep])
            # labels.append(torch.LongTensor(len(box[keep])).fill_(i+1))
            labels.append(torch.full_like(score[keep], fill_value=i+1, dtype=torch.long, device=device))
            # labels.append(torch.full(size=torch.Size([score[keep].size()[0]]), fill_value=i+1, dtype=torch.long,
            #                          device=device))

            scores.append(score[keep])

        if not boxes:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)
        return boxes, labels, scores


def change_box_order(boxes, order):
    """Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    """
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a], 1)
    return torch.cat([a-b/2,a+b/2], 1)


def box_clamp(boxes, xmin, ymin, xmax, ymax):
    """Clamp boxes.

    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) clamped boxes.
    """
    boxes[:,0].clamp_(min=xmin, max=xmax)
    boxes[:,1].clamp_(min=ymin, max=ymax)
    boxes[:,2].clamp_(min=xmin, max=xmax)
    boxes[:,3].clamp_(min=ymin, max=ymax)
    return boxes


def box_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    # N = box1.size(0)
    # M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou
