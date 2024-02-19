import torchvision
from mmengine.dataset import DefaultSampler

from diffengine.datasets import HFDataset
from diffengine.datasets.transforms import (
    CLIPImageProcessor,
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    RandomTextDrop,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import SDCheckpointHook, VisualizationHook

train_pipeline = [
    dict(type=CLIPImageProcessor),
    dict(type=RandomTextDrop, p=0.05),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Resize,
         size=768, interpolation="bilinear"),
    dict(type=RandomCrop, size=768),
    dict(type=RandomHorizontalFlip, p=0.5),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize,
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]),
    dict(type=PackInputs, input_keys=["img", "text", "clip_img"]),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=HFDataset,
        dataset="lambdalabs/pokemon-blip-captions",
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=SDCheckpointHook),
]
