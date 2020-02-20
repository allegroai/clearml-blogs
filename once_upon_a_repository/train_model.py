import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import chain
from operator import add

import numpy as np
import torch
from PIL import Image
from ignite.engine import Events
from pathlib2 import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.coco import CocoDetection
from trains import Task

from engines import create_trainer, create_evaluator
from torchvision_references import utils
from torchvision_references.coco_eval import CocoEvaluator
from torchvision_references.coco_utils import convert_to_coco_api
from transforms import get_transform
from utilities import draw_debug_images, draw_mask, get_model_instance_segmentation, safe_collate, get_iou_types

task = Task.init(project_name='Object Detection with TRAINS, Ignite and TensorBoard',
                 task_name='Train MaskRCNN with torchvision')

configuration_data = {'image_size': 512, 'mask_predictor_hidden_layer': 256}
configuration_data = task.connect_configuration(configuration_data)


class CocoMask(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, use_mask=True):
        super(CocoMask, self).__init__(root, annFile, transforms, target_transform, transform)
        self.transforms = transforms
        self.use_mask = use_mask
    
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        if len(ann_ids) == 0:
            return None
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        # From boxes [x, y, w, h] to [x1, y1, x2, y2]
        new_target = {"image_id": torch.as_tensor(target[0]['image_id'], dtype=torch.int64),
                      "area": torch.as_tensor([obj['area'] for obj in target], dtype=torch.float32),
                      "iscrowd": torch.as_tensor([obj['iscrowd'] for obj in target], dtype=torch.int64),
                      "boxes": torch.as_tensor([obj['bbox'][:2] + list(map(add, obj['bbox'][:2], obj['bbox'][2:]))
                                                for obj in target], dtype=torch.float32),
                      "labels": torch.as_tensor([obj['category_id'] for obj in target], dtype=torch.int64)}
        if self.use_mask:
            mask = [coco.annToMask(ann) for ann in target]
            if len(mask) > 1:
                mask = np.stack(tuple(mask), axis=0)
            new_target["masks"] = torch.as_tensor(mask, dtype=torch.uint8)
        
        if self.transforms is not None:
            img, new_target = self.transforms(img, new_target)
        
        return img, new_target


def get_data_loaders(train_ann_file, test_ann_file, batch_size, test_size, image_size, use_mask):
    # first, crate PyTorch dataset objects, for the train and validation data.
    dataset = CocoMask(
        root=Path.joinpath(Path(train_ann_file).parent.parent, train_ann_file.split('_')[1].split('.')[0]),
        annFile=train_ann_file,
        transforms=get_transform(train=True, image_size=image_size),
        use_mask=use_mask)
    dataset_test = CocoMask(
        root=Path.joinpath(Path(test_ann_file).parent.parent, test_ann_file.split('_')[1].split('.')[0]),
        annFile=test_ann_file,
        transforms=get_transform(train=False, image_size=image_size),
        use_mask=use_mask)
    
    labels_enumeration = dataset.coco.cats
    
    indices_val = torch.randperm(len(dataset_test)).tolist()
    dataset_val = torch.utils.data.Subset(dataset_test, indices_val[:test_size])

    # set train and validation data-loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                              collate_fn=safe_collate, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=6,
                            collate_fn=safe_collate, pin_memory=True)
    
    return train_loader, val_loader, labels_enumeration


def run(task_args):
    # Define train and test datasets
    train_loader, val_loader, labels_enum = get_data_loaders(task_args.train_dataset_ann_file,
                                                             task_args.val_dataset_ann_file,
                                                             task_args.batch_size,
                                                             task_args.test_size,
                                                             configuration_data.get('image_size'),
                                                             use_mask=True)
    val_dataset = list(chain.from_iterable(zip(*batch) for batch in iter(val_loader)))
    coco_api_val_dataset = convert_to_coco_api(val_dataset)
    num_classes = max(labels_enum.keys()) + 1  # number of classes plus one for background class
    configuration_data['num_classes'] = num_classes
    
    # Set the training device to GPU if available - if not set it to CPU
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False  # optimization for fixed input size
    
    model = get_model_instance_segmentation(num_classes, configuration_data.get('mask_predictor_hidden_layer'))
    iou_types = get_iou_types(model)
    
    # if there is more than one GPU, parallelize the model
    if torch.cuda.device_count() > 1:
        print("{} GPUs were detected - we will use all of them".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    
    # copy the model to each device
    model.to(device)
    
    if task_args.input_checkpoint:
        print('Loading model checkpoint from '.format(task_args.input_checkpoint))
        input_checkpoint = torch.load(task_args.input_checkpoint, map_location=torch.device(device))
        model.load_state_dict(input_checkpoint['model'])
    
    writer = SummaryWriter(log_dir=task_args.log_dir)
    
    # define Ignite's train and evaluation engine
    trainer = create_trainer(model, device)
    evaluator = create_evaluator(model, device)
    
    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        engine.state.optimizer = torch.optim.SGD(params,
                                                 lr=task_args.lr,
                                                 momentum=task_args.momentum,
                                                 weight_decay=task_args.weight_decay)
        engine.state.scheduler = torch.optim.lr_scheduler.StepLR(engine.state.optimizer, step_size=3, gamma=0.1)
        if task_args.input_checkpoint and task_args.load_optimizer:
            engine.state.optimizer.load_state_dict(input_checkpoint['optimizer'])
            engine.state.scheduler.load_state_dict(input_checkpoint['lr_scheduler'])
    
    @trainer.on(Events.EPOCH_STARTED)
    def on_epoch_started(engine):
        model.train()
        engine.state.warmup_scheduler = None
        if engine.state.epoch == 1:
            warmup_iters = min(task_args.warmup_iterations, len(train_loader) - 1)
            print('Warm up period was set to {} iterations'.format(warmup_iters))
            warmup_factor = 1. / warmup_iters
            engine.state.warmup_scheduler = utils.warmup_lr_scheduler(engine.state.optimizer, warmup_iters, warmup_factor)
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def on_iteration_completed(engine):
        images, targets, loss_dict_reduced = engine.state.output
        if engine.state.iteration % task_args.log_interval == 0:
            loss = sum(loss for loss in loss_dict_reduced.values()).item()
            print("Epoch: {}, Iteration: {}, Loss: {}".format(engine.state.epoch, engine.state.iteration, loss))
            for k, v in loss_dict_reduced.items():
                writer.add_scalar("loss/{}".format(k), v.item(), engine.state.iteration)
            writer.add_scalar("loss/total_loss", sum(loss for loss in loss_dict_reduced.values()).item(), engine.state.iteration)
            writer.add_scalar("learning rate/lr", engine.state.optimizer.param_groups[0]['lr'], engine.state.iteration)
        
        if engine.state.iteration % task_args.debug_images_interval == 0:
            for n, debug_image in enumerate(draw_debug_images(images, targets)):
                writer.add_image("training/image_{}".format(n), debug_image, engine.state.iteration, dataformats='HWC')
                if 'masks' in targets[n]:
                    writer.add_image("training/image_{}_mask".format(n),
                                     draw_mask(targets[n]), engine.state.iteration, dataformats='HW')
        images = targets = loss_dict_reduced = engine.state.output = None
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        engine.state.scheduler.step()
        evaluator.run(val_loader)
        for res_type in evaluator.state.coco_evaluator.iou_types:
            average_precision_05 = evaluator.state.coco_evaluator.coco_eval[res_type].stats[1]
            writer.add_scalar("validation-{}/average precision 0_5".format(res_type), average_precision_05,
                              engine.state.iteration)
        checkpoint_path = os.path.join(task_args.output_dir, 'model_epoch_{}.pth'.format(engine.state.epoch))
        print('Saving model checkpoint')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': engine.state.optimizer.state_dict(),
            'lr_scheduler': engine.state.scheduler.state_dict(),
            'epoch': engine.state.epoch,
            'configuration': configuration_data,
            'labels_enumeration': labels_enum}
        utils.save_on_master(checkpoint, checkpoint_path)
        print('Model checkpoint from epoch {} was saved at {}'.format(engine.state.epoch, checkpoint_path))
        evaluator.state = checkpoint = None

    @evaluator.on(Events.STARTED)
    def on_evaluation_started(engine):
        model.eval()
        engine.state.coco_evaluator = CocoEvaluator(coco_api_val_dataset, iou_types)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def on_eval_iteration_completed(engine):
        images, targets, results = engine.state.output
        if engine.state.iteration % task_args.log_interval == 0:
            print("Evaluation: Iteration: {}".format(engine.state.iteration))
        
        if engine.state.iteration % task_args.debug_images_interval == 0:
            for n, debug_image in enumerate(draw_debug_images(images, targets, results)):
                writer.add_image("evaluation/image_{}_{}".format(engine.state.iteration, n),
                                 debug_image, trainer.state.iteration, dataformats='HWC')
                if 'masks' in targets[n]:
                    writer.add_image("evaluation/image_{}_{}_mask".format(engine.state.iteration, n),
                                     draw_mask(targets[n]), trainer.state.iteration, dataformats='HW')
                    curr_image_id = int(targets[n]['image_id'])
                    writer.add_image("evaluation/image_{}_{}_predicted_mask".format(engine.state.iteration, n),
                                     draw_mask(results[curr_image_id]).squeeze(), trainer.state.iteration, dataformats='HW')
        images = targets = results = engine.state.output = None

    @evaluator.on(Events.COMPLETED)
    def on_evaluation_completed(engine):
        # gather the stats from all processes
        engine.state.coco_evaluator.synchronize_between_processes()
        
        # accumulate predictions from all images
        engine.state.coco_evaluator.accumulate()
        engine.state.coco_evaluator.summarize()

    trainer.run(train_loader, max_epochs=task_args.epochs)
    writer.close()
    
    
if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--warmup_iterations', type=int, default=5000,
                        help='Number of iteration for warmup period (until reaching base learning rate)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training and validation')
    parser.add_argument('--test_size', type=int, default=2000,
                        help='number of frames from the test dataset to use for validation')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--debug_images_interval', type=int, default=500,
                        help='how many batches to wait before logging debug images')
    parser.add_argument('--train_dataset_ann_file', type=str,
                        default='~/bigdata/coco/annotations/instances_train2017.json',
                        help='annotation file of train dataset')
    parser.add_argument('--val_dataset_ann_file', type=str, default='~/bigdata/coco/annotations/instances_val2017.json',
                        help='annotation file of test dataset')
    parser.add_argument('--input_checkpoint', type=str, default='',
                        help='Loading model weights from this checkpoint.')
    parser.add_argument('--load_optimizer', default=False, type=bool,
                        help='Use optimizer and lr_scheduler saved in the input checkpoint to resume training')
    parser.add_argument("--output_dir", type=str, default="/tmp/checkpoints",
                        help="output directory for saving models checkpoints")
    parser.add_argument("--log_dir", type=str, default="/tmp/tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate for optimizer")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="weight decay for optimizer")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        utils.mkdir(args.output_dir)
    if not os.path.exists(args.log_dir):
        utils.mkdir(args.log_dir)

    run(args)
