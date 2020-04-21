import json

import attr
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.resnet import resnet50, resnet101, resnet152
from torchvision.models.vgg import vgg16
from torchvision.transforms import functional as F

from torchvision_references import utils


def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return utils.collate_fn(batch)


def draw_boxes(im, boxes, labels, color=(150, 0, 0)):
    for box, draw_label in zip(boxes, labels):
        draw_box = box.astype('int')
        im = cv2.rectangle(im, tuple(draw_box[:2]), tuple(draw_box[2:]), color, 2)
        im = cv2.putText(im, str(draw_label), (draw_box[0], max(0, draw_box[1]-5)),
                         cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
    return im


def draw_debug_images(images, targets, predictions=None, score_thr=0.3):
    debug_images = []
    for image, target in zip(images, targets):
        img = draw_boxes(np.array(F.to_pil_image(image.cpu())),
                         [box.cpu().numpy() for box in target['boxes']],
                         [label.item() for label in target['labels']])
        if predictions:
            img = draw_boxes(img,
                             [box.cpu().numpy() for box, score in
                              zip(predictions[target['image_id'].item()]['boxes'],
                                  predictions[target['image_id'].item()]['scores']) if score >= score_thr],
                             [label.item() for label, score in
                              zip(predictions[target['image_id'].item()]['labels'],
                                  predictions[target['image_id'].item()]['scores']) if score >= score_thr],
                             color=(0, 150, 0))
        debug_images.append(img)
    return debug_images


def draw_mask(target):
    masks = [channel*label for channel, label in zip(target['masks'].cpu().numpy(), target['labels'].cpu().numpy())]
    masks_sum = sum(masks)
    masks_out = masks_sum + 25*(masks_sum > 0)
    return (masks_out*int(255/masks_out.max())).astype('uint8')


def get_backbone(backbone_name):
    if backbone_name == 'vgg16':
        return vgg16(pretrained=True)
    elif backbone_name == 'resnet50':
        return resnet50(pretrained=True)
    elif backbone_name == 'resnet101':
        return resnet101(pretrained=True)
    elif backbone_name == 'resnet152':
        return resnet152(pretrained=True)
    else:
        raise ValueError('Only "vgg16", "resnet50", "resnet101" and "resnet152" are supported backbone names')


def get_model_instance_segmentation(num_classes, hidden_layer):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@attr.s(auto_attribs=True)
class CocoLikeAnnotations():
    def __attrs_post_init__(self):
        self.coco_like_json: dict = {'images': [], 'annotations': []}
        self._ann_id: int = 0

    def update_images(self, file_name, height, width, id):
        self.coco_like_json['images'].append({'file_name': file_name,
                                         'height': height, 'width': width,
                                         'id': id})

    def update_annotations(self, box, label_id, image_id, is_crowd=0):
        segmentation, bbox, area = self.extract_coco_info(box)
        self.coco_like_json['annotations'].append({'segmentation': segmentation, 'bbox': bbox, 'area': area,
                                              'category_id': int(label_id), 'id': self._ann_id, 'iscrowd': is_crowd,
                                              'image_id': image_id})
        self._ann_id += 1

    @staticmethod
    def extract_coco_info(box):
        segmentation = list(map(int, [box[0], box[1], box[0], box[3], box[2], box[3], box[2], box[1]]))
        bbox = list(map(int, np.append(box[:2], (box[2:] - box[:2]))))
        area = int(bbox[2] * bbox[3])
        return segmentation, bbox, area

    def dump_to_json(self, path_to_json='/tmp/inference_results/inference_results.json'):
        with open(path_to_json, "w") as write_file:
            json.dump(self.coco_like_json, write_file)
