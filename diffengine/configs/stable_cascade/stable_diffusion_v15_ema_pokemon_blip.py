from mmengine.config import read_base

from diffengine.engine.hooks import SDCheckpointHook, UnetEMAHook, VisualizationHook

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_cascade import *
    from .._base_.schedules.schedule_50e import *


custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=SDCheckpointHook),
    dict(type=UnetEMAHook, momentum=1e-4, priority="ABOVE_NORMAL"),
]
