import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    # LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import torch.nn as nn
from diffusers import DDIMScheduler



def init_models(model_id, data_type):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=data_type)
    unet = pipe.unet
    vae = pipe.vae
    clip = pipe.text_encoder
    clip_tokenizer = pipe.tokenizer
    return unet, vae, clip, clip_tokenizer, pipe


class DenoiseImgOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        gt_np (`np.ndarray`):
            Predicted gt map, with values in the range of [0, 1].
    """

    gt_np: np.ndarray


class DenoiseImgPipeline(nn.Module):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        low_resources (`bool`, *optional*):
            Setting up the environment for computing resources.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    rgb_latent_scale_factor = 0.18215

    def __init__(
            self,
            model_id,
            data_type,
            low_resources: Optional[bool] = True,
            default_denoising_steps: Optional[int] = None,
            default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.unet, self.vae, self.text_encoder, self.tokenizer, self.pipe = init_models(
            model_id=model_id,
            data_type=data_type)

        self.low_resources = low_resources
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None
        # Trainability
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        # self.unet.enable_xformers_memory_efficient_attention()


    def get_text_embedding_(self, prompt, device, negative_prompt=None,
                            do_classifier_free_guidance=True, num_images_per_prompt=1):
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        return negative_prompt_embeds, prompt_embeds

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    def norm_img_to_latent(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode norm RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`): range [-1,1]
                Input norm RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        # ## other way
        # latents = self.vae.encode(rgb_in).latent_dist.sample(generator=None) * self.rgb_latent_scale_factor
        return rgb_latent

    def latent_to_img_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent into img.

        Args:
            latent (`torch.Tensor`):
                latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded img range [0,1].
        """
        # scale latent
        latent = latent / self.rgb_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(latent)
        img_norm = self.vae.decoder(z)
        img = (img_norm / 2 + 0.5).clamp(0, 1)
        return img

    def img_tensor_to_image(self, img_norm: torch.Tensor) -> Image:
        '''

        Args:
            img_norm (`torch.Tensor`): range [0,1]

        Returns:
            PIL.Image
        '''
        images = img_norm.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def img_process_(self, img):
        '''

        Args:
            img: tensor - [B, 3, H, W] - range [0,255]

        Returns:
            batch_size: int
            img_norm - [B, 3, H, W] - range [-1,1]

        '''
        batch_size = img.shape[0]
        img_norm = img / 255.0 * 2.0 - 1.0
        return img_norm, batch_size
