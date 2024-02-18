from diffusers import DDPMWuerstchenScheduler
from diffusers.pipelines.stable_cascade.modeling_stable_cascade_common import (
    StableCascadeUnet,
)
from diffusers.pipelines.wuerstchen.modeling_paella_vq_model import (
    PaellaVQModel,
)
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from diffengine.models.editors import StableCascade
from diffengine.models.editors.stable_cascade.effnet import EfficientNetEncoder

base_model = "stabilityai/stable-cascade"
model = dict(type=StableCascade,
             model=base_model,
             tokenizer=dict(type=CLIPTokenizer.from_pretrained,
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMWuerstchenScheduler),
             text_encoder=dict(type=CLIPTextModelWithProjection.from_pretrained,
                               subfolder="text_encoder"),
             vqgan=dict(
                type=PaellaVQModel.from_pretrained,
                subfolder="vqgan"),
             decoder=dict(type=StableCascadeUnet.from_pretrained,
                             subfolder="decoder"),
             effnet=dict(type=EfficientNetEncoder))
