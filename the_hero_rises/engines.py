import math
import copy
import torch

from ignite.engine import Engine
from torchvision_references import utils


def create_trainer(model, device):
    def update_model(engine, batch):
        images, targets = copy.deepcopy(batch)
        images_model, targets_model = prepare_batch(batch, device=device)

        loss_dict = model(images_model, targets_model)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        engine.state.optimizer.zero_grad()
        if not math.isfinite(loss_value):
            print("Loss is {}, resetting loss and skipping training iteration".format(loss_value))
            print('Loss values were: ', loss_dict_reduced)
            print('Input labels were: ', [target['labels'] for target in targets])
            print('Input boxes were: ', [target['boxes'] for target in targets])
            loss_dict_reduced = {k: torch.tensor(0) for k, v in loss_dict_reduced.items()}
        else:
            losses.backward()
            engine.state.optimizer.step()

        if engine.state.warmup_scheduler is not None:
            engine.state.warmup_scheduler.step()

        images_model = targets_model = None

        return images, targets, loss_dict_reduced
    return Engine(update_model)


def create_evaluator(model, device):
    def update_model(engine, batch):
        images, targets = prepare_batch(batch, device=device)
        images_model = copy.deepcopy(images)

        torch.cuda.synchronize()
        with torch.no_grad():
            outputs = model(images_model)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        engine.state.coco_evaluator.update(res)

        images_model = outputs = None

        return images, targets, res
    return Engine(update_model)


def prepare_batch(batch, device=None):
    images, targets = batch
    images = list(image.to(device, non_blocking=True) for image in images)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return images, targets
