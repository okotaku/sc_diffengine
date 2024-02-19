from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_cascade import *
    from .._base_.schedules.schedule_50e import *
