from diffusers import DDPMWuerstchenScheduler
from diffusers.pipelines.stable_cascade.modeling_stable_cascade_common import (
    StableCascadeUnet,
)
from diffusers.pipelines.wuerstchen.modeling_paella_vq_model import (
    PaellaVQModel,
)
from transformers import (
    CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
)

from diffengine.models.editors import StableCascade
from diffengine.models.editors.stable_cascade.effnet import EfficientNetEncoder

base_model = "stabilityai/stable-cascade"
prior_model = "stabilityai/stable-cascade-prior"
model = dict(type=StableCascade,
             model=base_model,
             prior_model=prior_model,
             tokenizer=dict(type=CLIPTokenizer.from_pretrained,
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMWuerstchenScheduler),
             text_encoder=dict(type=CLIPTextModelWithProjection.from_pretrained,
                               subfolder="text_encoder"),
             image_encoder=dict(
                 type=CLIPVisionModelWithProjection.from_pretrained,
                 subfolder="image_encoder"),
             prior=dict(type=StableCascadeUnet.from_pretrained,
                             subfolder="prior"),
             effnet=dict(type=EfficientNetEncoder))
