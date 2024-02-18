from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
from mmengine import print_log
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from peft import get_peft_model
from torch import nn

from diffengine.models.archs import create_peft_config
from diffengine.models.editors.stable_cascade.data_preprocessor import (
    SDDataPreprocessor,
)
from diffengine.models.losses import L2Loss
from diffengine.models.utils import WhiteNoise, WuerstchenRandomTimeSteps


class StableCascade(BaseModel):
    """Stable Cascade.

    Args:
    ----
        tokenizer (dict): Config of tokenizer.
        scheduler (dict): Config of scheduler.
        text_encoder (dict): Config of text encoder.
        vqgan (dict): Config of vqgan.
        decoder (dict): Config of decoder.
        effnet (dict): Config of effnet.
        prior_model (str): pretrained model name of stable cascade.
            Defaults to '"stabilityai/stable-cascade-prior"'.
        model (str): pretrained model name of stable cascade.
            Defaults to 'stabilityai/stable-cascade'.
        loss (dict): Config of loss. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0)``.
        decoder_lora_config (dict, optional): The LoRA config dict for Decoder.
            example. dict(type="LoRA", r=4). `type` is chosen from `LoRA`,
            `LoHa`, `LoKr`. Other config are same as the config of PEFT.
            https://github.com/huggingface/peft
            Defaults to None.
        text_encoder_lora_config (dict, optional): The LoRA config dict for
            Text Encoder. example. dict(type="LoRA", r=4). `type` is chosen
            from `LoRA`, `LoHa`, `LoKr`. Other config are same as the config of
            PEFT. https://github.com/huggingface/peft
            Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDDataPreprocessor`.
        noise_generator (dict, optional): The noise generator config.
            Defaults to ``dict(type='WhiteNoise')``.
        timesteps_generator (dict, optional): The timesteps generator config.
            Defaults to ``dict(type='TimeSteps')``.
        input_perturbation_gamma (float): The gamma of input perturbation.
            The recommended value is 0.1 for Input Perturbation.
            Defaults to 0.0.
        vae_batch_size (int): The batch size of vae. Defaults to 8.
        zeros_image_embeddings_prob (float): The probabilities to
            generate zeros image embeddings. Defaults to 0.1.
        finetune_text_encoder (bool, optional): Whether to fine-tune text
            encoder. Defaults to False.
        gradient_checkpointing (bool): Whether or not to use gradient
            checkpointing to save memory at the expense of slower backward
            pass. Defaults to False.
        enable_xformers (bool): Whether or not to enable memory efficient
            attention. Defaults to False.

    """

    def __init__(  # noqa: PLR0913
        self,
        tokenizer: dict,
        scheduler: dict,
        text_encoder: dict,
        vqgan: dict,
        decoder: dict,
        effnet: dict,
        prior_model: str = "stabilityai/stable-cascade-prior",
        model: str = "stabilityai/stable-cascade",
        loss: dict | None = None,
        decoder_lora_config: dict | None = None,
        text_encoder_lora_config: dict | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        vae_batch_size: int = 8,
        zeros_image_embeddings_prob: float = 0.1,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": SDDataPreprocessor}
        if loss is None:
            loss = {}
        if noise_generator is None:
            noise_generator = {}
        if timesteps_generator is None:
            timesteps_generator = {}
        super().__init__(data_preprocessor=data_preprocessor)
        if (
            decoder_lora_config is not None) and (
                text_encoder_lora_config is not None) and (
                    not finetune_text_encoder):
                print_log(
                    "You are using LoRA for Decoder and text encoder. "
                    "But you are not set `finetune_text_encoder=True`. "
                    "We will set `finetune_text_encoder=True` for you.")
                finetune_text_encoder = True
        if text_encoder_lora_config is not None:
            assert finetune_text_encoder, (
                "If you want to use LoRA for text encoder, "
                "you should set finetune_text_encoder=True."
            )
        if finetune_text_encoder and decoder_lora_config is not None:
            assert text_encoder_lora_config is not None, (
                "If you want to finetune text encoder with LoRA Decoder, "
                "you should set text_encoder_lora_config."
            )

        self.prior_model = prior_model
        self.model = model
        self.decoder_lora_config = deepcopy(decoder_lora_config)
        self.text_encoder_lora_config = deepcopy(text_encoder_lora_config)
        self.finetune_text_encoder = finetune_text_encoder
        self.gradient_checkpointing = gradient_checkpointing
        self.input_perturbation_gamma = input_perturbation_gamma
        self.enable_xformers = enable_xformers
        self.vae_batch_size = vae_batch_size
        self.zeros_image_embeddings_prob = zeros_image_embeddings_prob

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(
                loss,
                default_args={"type": L2Loss, "loss_weight": 1.0})
        self.loss_module: nn.Module = loss

        self.tokenizer = MODELS.build(
            tokenizer,
            default_args={"pretrained_model_name_or_path": model})
        self.scheduler = MODELS.build(
            scheduler,
            default_args={"pretrained_model_name_or_path": model})

        self.text_encoder = MODELS.build(
            text_encoder,
            default_args={"pretrained_model_name_or_path": model})
        self.vqgan = MODELS.build(
            vqgan,
            default_args={"pretrained_model_name_or_path": model})
        self.decoder = MODELS.build(
            decoder,
            default_args={"pretrained_model_name_or_path": model})
        self.effnet = MODELS.build(effnet)
        self.noise_generator = MODELS.build(
            noise_generator,
            default_args={"type": WhiteNoise})
        self.timesteps_generator = MODELS.build(
            timesteps_generator,
            default_args={"type": WuerstchenRandomTimeSteps})
        self.prepare_model()
        self.set_lora()
        self.set_xformers()

    def set_lora(self) -> None:
        """Set LORA for model."""
        if self.text_encoder_lora_config is not None:
            text_encoder_lora_config = create_peft_config(
                self.text_encoder_lora_config)
            self.text_encoder = get_peft_model(
                self.text_encoder, text_encoder_lora_config)
            self.text_encoder.print_trainable_parameters()

        if self.decoder_lora_config is not None:
            decoder_lora_config = create_peft_config(self.decoder_lora_config)
            self.decoder = get_peft_model(self.decoder, decoder_lora_config)
            self.decoder.print_trainable_parameters()

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.gradient_checkpointing:
            self.decoder.enable_gradient_checkpointing()
            if self.finetune_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        self.vqgan.requires_grad_(requires_grad=False)
        print_log("Set VQGAN untrainable.", "current")
        if not self.finetune_text_encoder:
            self.text_encoder.requires_grad_(requires_grad=False)
            print_log("Set Text Encoder untrainable.", "current")

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                self.decoder.enable_xformers_memory_efficient_attention()
            else:
                msg = "Please install xformers to enable memory efficient attention."
                raise ImportError(
                    msg,
                )

    @property
    def device(self) -> torch.device:
        """Get device information.

        Returns
        -------
            torch.device: device.

        """
        return next(self.parameters()).device

    def train(self, *, mode: bool = True) -> None:
        """Convert the model into training mode."""
        super().train(mode)
        self.image_encoder.eval()
        if not self.finetune_text_encoder:
            self.text_encoder.eval()

    @torch.no_grad()
    def infer(self,
              prompt: list[str],
              negative_prompt: str | None = None,
              height: int = 1024,
              width: int = 1024,
              **kwargs) -> list[np.ndarray]:
        """Inference function.

        Args:
        ----
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`Optional[str]`):
                The prompt or prompts to guide the image generation.
                Defaults to None.
            height (int, optional):
                The height in pixels of the generated image. Defaults to None.
            width (int, optional):
                The width in pixels of the generated image. Defaults to None.
            **kwargs: Other arguments.

        """
        prior = StableCascadePriorPipeline.from_pretrained(
            self.prior_model,
            torch_dtype=(torch.bloat16 if self.device != torch.device("cpu")
                         else torch.float32),
        )
        decoder = StableCascadeDecoderPipeline.from_pretrained(
            self.model,
            vqgan=self.vqgan,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            decoder=self.decoder,
            torch_dtype=(torch.float16 if self.device != torch.device("cpu")
                         else torch.float32),
        )
        prior.set_progress_bar_config(disable=True)
        decoder.set_progress_bar_config(disable=True)
        images = []
        for p in prompt:
            prior_output = prior(
                prompt=prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                guidance_scale=4.0,
                num_inference_steps=20,
            )
            image = decoder(
                image_embeddings=prior_output.image_embeddings,
                prompt=p,
                negative_prompt=negative_prompt,
                num_inference_steps=10,
                guidance_scale=0.0,
                output_type="pil",
                **kwargs).images[0]
            images.append(np.array(image))

        del prior, decoder
        torch.cuda.empty_cache()

        return images

    def val_step(
            self,
            data: Union[tuple, dict, list]  # noqa
    ) -> list:
        """Val step."""
        msg = "val_step is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def test_step(
            self,
            data: Union[tuple, dict, list]  # noqa
    ) -> list:
        """Test step."""
        msg = "test_step is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def loss(self,
             model_pred: torch.Tensor,
             noise: torch.Tensor,
             timesteps: torch.Tensor,
             weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Calculate loss."""
        loss_dict = {}
        # calculate loss in FP32
        if self.loss_module.use_snr:
            loss = self.loss_module(
                model_pred.float(),
                noise.float(),
                timesteps,
                self.scheduler._alpha_cumprod(timesteps, self.device),  # noqa
                "epsilon",
                weight=weight)
        else:
            loss = self.loss_module(
                model_pred.float(), noise.float(), weight=weight)
        loss_dict["loss"] = loss
        return loss_dict

    def _preprocess_model_input(self,
                                latents: torch.Tensor,
                                noise: torch.Tensor,
                                timesteps: torch.Tensor) -> torch.Tensor:
        """Preprocess model input."""
        if self.input_perturbation_gamma > 0:
            input_noise = noise + self.input_perturbation_gamma * torch.randn_like(
                noise)
        else:
            input_noise = noise
        return self.scheduler.add_noise(latents, input_noise, timesteps)

    def _forward_vae(self, img: torch.Tensor, num_batches: int,
                     ) -> torch.Tensor:
        """Forward vae."""
        latents = [
            self.vae.encode(
                img[i : i + self.vae_batch_size],
            ).latents for i in range(
                0, num_batches, self.vae_batch_size)
        ]
        return torch.cat(latents, dim=0)

    def forward(
            self,
            inputs: dict,
            data_samples: Optional[list] = None,  # noqa
            mode: str = "loss") -> dict:
        """Forward function.

        Args:
        ----
            inputs (dict): The input dict.
            data_samples (Optional[list], optional): The data samples.
                Defaults to None.
            mode (str, optional): The mode. Defaults to "loss".

        Returns:
        -------
            dict: The loss dict.

        """
        assert mode == "loss"
        num_batches = len(inputs["img"])

        inputs_text = self.tokenizer(
            inputs["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt")

        latents = self._forward_vae(inputs["img"], num_batches)

        noise = self.noise_generator(latents)

        timesteps = self.timesteps_generator(num_batches,
                                            self.device)

        noisy_latents = self._preprocess_model_input(latents, noise, timesteps)

        text_encoder_output = self.text_encoder(
            inputs_text.input_ids.to(self.device),
            attention_mask=inputs_text.attention_mask.to(self.device))
        prompt_embeds_pooled = text_encoder_output.text_embeds.unsqueeze(1)

        image_embeds = self.effnet(inputs["effnet"])
        # random zeros image embeddings
        mask = torch.multinomial(
            torch.Tensor([
                self.zeros_image_embeddings_prob,
                1 - self.zeros_image_embeddings_prob,
            ]),
            len(image_embeds),
            replacement=True).to(image_embeds)
        image_embeds = (image_embeds * mask.view(-1, 1)).view(num_batches, 1, 1, -1)

        model_pred = self.decoder(
            noisy_latents,
            timesteps,
            clip_text_pooled=prompt_embeds_pooled).sample

        return self.loss(model_pred, noise, timesteps)
