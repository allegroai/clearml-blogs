"""
SSD model on top of TorchVision feature extractor.
The constant values are suitable to a 512X512 image. Automatic change to a different image size
can be done by runnint the dry_run method.

requirements: PyTorch and TorchVision

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from SSD.box_coder import SSDBoxCoder

# Aspect ration between current layer and original image size.
# I.e, how many pixel steps on the original image are equivalent to a single pixel step on the feature map.
STEPS = (8, 16, 32, 64, 128, 256, 512)
# Length of the shorter anchor rectangle face sizes, for each feature map.
BOX_SIZES = (35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6)
# Aspect ratio of the rectanglar SSD anchors, besides 1:1
ASPECT_RATIOS = ((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,))
# feature maps sizes.
FM_SIZES = (64, 32, 16, 8, 4, 2, 1)
# Amount of anchors for each feature map
NUM_ANCHORS = (4, 6, 6, 6, 6, 4, 4)
# Amount of each feature map channels, i.e third dimension.
IN_CHANNELS = (512, 1024, 512, 256, 256, 256, 256)


class HeadsExtractor(nn.Module):
    def __init__(self, backbone):
        super(HeadsExtractor, self).__init__()
        
        def split_backbone(net):
            features_extraction = [x for x in net.children()][:-2]
            
            if type(net) == torchvision.models.vgg.VGG:
                features_extraction = [*features_extraction[0]]
                net_till_conv4_3 = features_extraction[:-8]
                rest_of_net = features_extraction[-7:-1]
            elif type(net) == torchvision.models.resnet.ResNet:
                net_till_conv4_3 = features_extraction[:-2]
                rest_of_net = features_extraction[-2]
            else:
                raise ValueError('We only support VGG and ResNet backbones')
            return nn.Sequential(*net_till_conv4_3), nn.Sequential(*rest_of_net)
        
        self.till_conv4_3, self.till_conv5_3 = split_backbone(backbone)
        self.norm4 = L2Norm(512, 20)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        
        self.conv12_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv12_2 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
    
    def forward(self, x):
        hs = []
        h = self.till_conv4_3(x)
        hs.append(self.norm4(h))
        
        if type(self.till_conv5_3[-1]) != torchvision.models.resnet.Bottleneck:
            h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)
            h = self.till_conv5_3(h)
            h = F.max_pool2d(h, kernel_size=3, stride=1, padding=1, ceil_mode=True)
            
            h = F.relu(self.conv6(h))
            h = F.relu(self.conv7(h))
        else:
            h = self.till_conv5_3(h)
        hs.append(h)  # conv7
        
        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)  # conv8_2
        
        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)  # conv9_2
        
        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)  # conv10_2
        
        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)  # conv11_2
        
        h = F.relu(self.conv12_1(h))
        h = F.relu(self.conv12_2(h))
        hs.append(h)  # conv12_2
        return hs
    
    
class SSD(nn.Module):
    def __init__(self, backbone, num_classes, loss_function,
                 num_anchors=NUM_ANCHORS,
                 in_channels=IN_CHANNELS,
                 steps=STEPS,
                 box_sizes=BOX_SIZES,
                 aspect_ratios=ASPECT_RATIOS,
                 fm_sizes=FM_SIZES,
                 heads_extractor_class=HeadsExtractor):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_channels = in_channels
        self.steps = steps
        self.box_sizes = box_sizes
        self.aspect_ratios = aspect_ratios
        self.fm_sizes = fm_sizes

        self.extractor = heads_extractor_class(backbone)
        self.criterion = loss_function
        self.box_coder = SSDBoxCoder(self.steps, self.box_sizes, self.aspect_ratios, self.fm_sizes)

        self._create_heads()

    def _create_heads(self):
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i] * 4, kernel_size=3, padding=1)]
            self.cls_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i] * self.num_classes, kernel_size=3,
                                          padding=1)]

    def change_input_size(self, x):
        heads = self.extractor(x)
        self.fm_sizes = tuple([head.shape[-1] for head in heads])
        image_size = x.shape[-1]
        self.steps = tuple([image_size//fm for fm in self.fm_sizes])
        self.box_coder = SSDBoxCoder(self.steps, self.box_sizes, self.aspect_ratios, self.fm_sizes)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        loc_preds = []
        cls_preds = []
        input_images = torch.stack(images) if isinstance(images, list) else images
        extracted_batch = self.extractor(input_images)
        for i, x in enumerate(extracted_batch):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)

        if self.training:
            encoded_targets = [self.box_coder.encode(target['boxes'], target['labels']) for target in targets]
            loc_targets = torch.stack([encoded_target[0] for encoded_target in encoded_targets])
            cls_targets = torch.stack([encoded_target[1] for encoded_target in encoded_targets])
            losses = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            return losses

        detections = []
        for batch, (loc, cls) in enumerate(zip(loc_preds.split(split_size=1, dim=0),
                                               cls_preds.split(split_size=1, dim=0))):
            boxes, labels, scores = self.box_coder.decode(loc.squeeze(), F.softmax(cls.squeeze(), dim=1))
            detections.append({'boxes': boxes, 'labels': labels, 'scores': scores})

        return detections


class L2Norm(nn.Module):
    """L2Norm layer across all channels."""

    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None, :, None, None]
        return scale * x



# Based on https://github.com/kuangliu/torchcv/tree/master/examples/ssd
