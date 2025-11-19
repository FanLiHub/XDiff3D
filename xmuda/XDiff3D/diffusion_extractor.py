# An official reimplemented version of Marigold training script.
# Last modified: 2024-08-16
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
import os
from typing import List, Union
import math
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.transformer_2d import Transformer2DModelOutput
from diffusers import DDPMScheduler
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from PIL import Image
import torch.nn as nn
import random
from .diffusion_component import DenoiseImgPipeline, DenoiseImgOutput
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
from agent_query import object_query_2D_Dual

def get_tv_resample_method(method_str: str) -> InterpolationMode:
    resample_method_dict = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
        # "nearest-exact": InterpolationMode.NEAREST_EXACT,
    }
    resample_method = resample_method_dict.get(method_str, None)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {resample_method}")
    else:
        return resample_method

def collect_feats(unet, mode, idxs, flag_layer='resnet'):
    feats = []
    layers = collect_layers(unet, mode, idxs, flag_layer)
    for module in layers:
        feats.append(module.feats)
        module.feats = {}
        module.timestep = None
    return feats


def collect_layers(unet, mode, idxs=None, flag_layer='resnet'):
    layers = []
    if flag_layer == 'resnet':
        if mode == 'up':
            for i, up_block in enumerate(unet.up_blocks):
                for j, module in enumerate(up_block.resnets):
                    if idxs is None or (i, j) in idxs:
                        layers.append(module)
        elif mode == 'down':
            for i, down_block in enumerate(unet.down_blocks):
                for j, module in enumerate(down_block.resnets):
                    if idxs is None or (i, j) in idxs:
                        layers.append(module)
    if flag_layer == 'attention':
        if mode == 'up':
            for i, up_block in enumerate(unet.up_blocks):
                if hasattr(up_block, 'attentions'):
                    for j, module in enumerate(up_block.attentions):
                        if idxs is None or (i, j) in idxs:
                            layers.append(module)
        elif mode == 'down':
            for i, down_block in enumerate(unet.down_blocks):
                if hasattr(down_block, 'attentions'):
                    for j, module in enumerate(down_block.attentions):
                        if idxs is None or (i, j) in idxs:
                            layers.append(module)
    return layers


def init_block_func(
        unet,
        mode,
        save_hidden=False,
        use_hidden=False,
        reset=True,
        save_timestep=[],
        idxs=[(1, 0)],
        flag_layer='resnet',
):
    def renet_new_forward(self, input_tensor, temb):
        # https://github.com/huggingface/diffusers/blob/ad9d7ce4763f8fb2a9e620bff017830c26086c36/src/diffusers/models/resnet.py#L372
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        if save_hidden:
            if save_timestep is None or self.timestep in save_timestep:
                self.feats[self.timestep] = hidden_states
        elif use_hidden:
            hidden_states = self.feats[self.timestep]
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor

    def attn_new_forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            timestep=None,
            class_labels=None,
            cross_attention_kwargs=None,
            return_dict: bool = True,
    ):
        # 1. Input
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = self.proj_in(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = self.proj_out(hidden_states)
            else:
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()
        elif self.is_input_patches:
            # TODO: cleanup!
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)

            # unpatchify
            height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        if save_hidden:
            if save_timestep is None or self.timestep in save_timestep:
                self.feats[self.timestep] = hidden_states

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    if 'resnet' in flag_layer:
        layers = collect_layers(unet, mode, idxs, flag_layer='resnet')
        for module in layers:
            module.forward = renet_new_forward.__get__(module, type(module))
            if reset:
                module.feats = {}
                module.timestep = None
    if 'attention' in flag_layer:
        layers = collect_layers(unet, mode, idxs, flag_layer='attention')
        for module in layers:
            module.forward = attn_new_forward.__get__(module, type(module))
            if reset:
                module.feats = {}
                module.timestep = None


def set_timestep(unet, layer_indexes, timestep=None):
    layers = collect_layers(unet, 'up', layer_indexes, 'resnet')
    for module in layers:
        module_name = type(module).__name__
        module.timestep = timestep

def resize_max_res(
        img: torch.Tensor,
        max_edge_resolution: int,
        resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image.
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, (new_height, new_width), resample_method, antialias=True)
    return resized_img


def generate_seed_sequence(
        initial_seed: int,
        length: int,
        min_val=-0x8000_0000_0000_0000,
        max_val=0xFFFF_FFFF_FFFF_FFFF,
):
    if initial_seed is None:
        logging.warning("initial_seed is None, reproducibility is not guaranteed")
    random.seed(initial_seed)

    seed_sequence = []

    for _ in range(length):
        seed = random.randint(min_val, max_val)

        seed_sequence.append(seed)

    return seed_sequence


class DiffusionTrainer_(nn.Module):
    rgb_latent_scale_factor = 0.18215

    def __init__(
            self,
            model_id,
            layer_flag='resnets',
            mode='down_blocks',
            layer_indexes=[[0, 2], [1, 2], [2, 2], [3, 2]],
            timestep_list=[50,150],
            guidance_scale=-1,
            num_inference_steps=50,
            data_type='fp32',
            features_flag=False,
            hook_flag=True,
            multi_res_noise_flag=False,
            strength=0.9,
            annealed=True,
            downscale_strategy='original',
            channel_modific=False,
            channel_num=4,
            seed=2024,
            low_resources=False
    ):
        super().__init__()
        self.seed: Union[int, None] = seed  # used to generate seed sequence, set to `None` to train w/o seeding
        self.timestep_list = timestep_list
        self.guidance_scale = guidance_scale
        self.strength = strength,
        self.annealed = annealed,
        self.downscale_strategy = downscale_strategy,
        self.low_resources = low_resources
        self.num_inference_steps=num_inference_steps

        if data_type == 'bf':
            self.dtype = torch.bfloat16
        elif data_type == 'fp16':
            self.dtype = torch.float16
        elif data_type == 'fp32':
            self.dtype = torch.float32

        self.diff = DenoiseImgPipeline(model_id, self.dtype)
        # self.scheduler=self.diff.pipe.scheduler
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            # prediction_type="v_prediction",
        )
        # Adapt input layers
        if channel_modific == True and channel_num != self.diff.unet.config["in_channels"]:
            self._replace_unet_conv_in()
        # Encode empty text prompt
        self.encode_empty_text()
        # self.empty_text_embed = self.empty_text_embed.detach().clone()

        # Training noise scheduler
        ## Can be replaced by self.scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(model_id, "scheduler")
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
                self.prediction_type == self.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Multi-resolution noise
        self.apply_multi_res_noise = multi_res_noise_flag
        if self.apply_multi_res_noise:
            self.mr_noise_strength = strength
            self.annealed_mr_noise = annealed
            self.mr_noise_downscale_strategy = downscale_strategy

        # Internal variables
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

        self.hook_flag = hook_flag
        self.features_flag=features_flag
        if self.features_flag==True:
            ## collect features
            self.layer_flag = layer_flag  ## resnets attentions
            self.mode = mode  ## up_blocks down_blocks
            self.layer_indexes = layer_indexes
            if self.hook_flag == True:
                ## hook for features
                self.layers = []
                for l in self.layer_indexes:
                    if self.mode == 'up_blocks':
                        if self.layer_flag == 'resnets':
                            self.diff.unet.up_blocks[l[0]].resnets[l[1]].register_forward_hook(
                                lambda m, _, o: self.layers.append(o))
                        elif self.layer_flag == 'attentions':
                            self.diff.unet.up_blocks[l[0]].attentions[l[1]].register_forward_hook(
                                lambda m, _, o: self.layers.append(o))
                    elif self.mode == 'down_blocks':
                        if self.layer_flag == 'resnets':
                            self.diff.unet.down_blocks[l[0]].resnets[l[1]].register_forward_hook(
                                lambda m, _, o: self.layers.append(o))
                        elif self.layer_flag == 'attentions':
                            self.diff.unet.down_blocks[l[0]].attentions[l[1]].register_forward_hook(
                                lambda m, _, o: self.layers.append(o))
            else:
                ## rewrite forward function
                init_block_func(self.diff.unet, 'up', save_hidden=True, reset=True, idxs=self.layer_indexes,
                                save_timestep=[0], flag_layer='resnet')

        # frozen params
        for name, params in self.diff.unet.named_parameters():
            params.requires_grad = False

        self.object_query_2D = object_query_2D_Dual(self.num_stages, len(layer_indexes), 32)

    def _replace_unet_conv_in(self):
        # replace the first layer to accept 8 in_channels
        _weight = self.diff.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.diff.unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight *= 0.5
        # new conv_in channel
        _n_convin_out_channel = self.diff.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.diff.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        self.diff.unet.config["in_channels"] = 8
        logging.info("Unet config is updated")
        return

    def feature_extraction(
            self,
            rgb_in: torch.Tensor,
            timestep_list: int,
            add_noise_flag=False,
            prompt: str = None,
    ) -> torch.Tensor:

        device = self.diff.text_encoder.device
        # globally consistent random generators
        if self.seed is not None:
            local_seed = self._get_next_seed()
            rand_num_generator = torch.Generator('cuda')
            rand_num_generator.manual_seed(local_seed)
        else:
            rand_num_generator = None

        rgb, batch_size = self.diff.img_process_(rgb_in)
        # Encode image
        rgb_latent = self.diff.norm_img_to_latent(rgb)

        if prompt == None:
            # Batched empty text embedding
            if self.empty_text_embed is None:
                self.encode_empty_text()
            text_embed = self.empty_text_embed.repeat(
                (rgb_latent.shape[0], 1, 1)
            ).to(device)  # [B, 2, 1024]
        else:
            negative_text_embed, text_embed = self.get_text_embedding_(prompt, self.diff.text_encoder.device)
            text_embed = torch.cat([negative_text_embed, text_embed])

        # Denoising list
        for t in timestep_list:
            if self.hook_flag == False:
                set_timestep(self.diff.unet, self.layer_indexes, t)

            # Sample a random timestep for each image
            t = torch.tensor(t).to(device=self.diff.unet.device).long()
            if add_noise_flag == True:
                ## add noise for latent
                # Sample noise
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        # calculate strength depending on t
                        strength = strength * (t / self.scheduler_timesteps)
                    noise = self.multi_res_noise_like(
                        rgb_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    noise = torch.randn(
                        rgb_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]

                # Add noise to the latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    rgb_latent, noise, t
                )  # [B, 4, h, w]
                unet_input = noisy_latents
            else:
                unet_input = rgb_latent

            # predict the noise residual
            noise_pred = self.predict_(unet_input, t, text_embed, prompt)

        if self.hook_flag == False:
            feats = collect_feats(self.diff.unet, 'up', idxs=self.layer_indexes, flag_layer='resnet')
            for s in range(len(feats)):
                self.object_query_2D.forward(feats, s, batch_first=False, has_cls_token=False)
            return self.object_query_2D.return_auto(feats)
        return



    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=100000,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def multi_res_noise_like(self, x, strength=0.9, downscale_strategy="original", generator=None):
        if torch.is_tensor(strength):
            strength = strength.reshape((-1, 1, 1, 1))
        b, c, w, h = x.shape

        device = x.device

        up_sampler = torch.nn.Upsample(size=(w, h), mode="bilinear")
        noise = torch.randn(x.shape, device=x.device, generator=generator)

        if "original" == downscale_strategy:
            for i in range(10):
                r = (
                        torch.rand(1, generator=generator, device=device) * 2 + 2
                )  # Rather than always going 2x,
                w, h = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
                noise += (
                        up_sampler(
                            torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                        )
                        * strength ** i
                )
                if w == 1 or h == 1:
                    break  # Lowest resolution is 1x1
        elif "every_layer" == downscale_strategy:
            for i in range(int(math.log2(min(w, h)))):
                w, h = max(1, int(w / 2)), max(1, int(h / 2))
                noise += (
                        up_sampler(
                            torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                        )
                        * strength ** i
                )
        elif "power_of_two" == downscale_strategy:
            for i in range(10):
                r = 2
                w, h = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
                noise += (
                        up_sampler(
                            torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                        )
                        * strength ** i
                )
                if w == 1 or h == 1:
                    break  # Lowest resolution is 1x1
        elif "random_step" == downscale_strategy:
            for i in range(10):
                r = (
                        torch.rand(1, generator=generator, device=device) * 2 + 2
                )  # Rather than always going 2x,
                w, h = max(1, int(w / (r))), max(1, int(h / (r)))
                noise += (
                        up_sampler(
                            torch.randn(b, c, w, h, generator=generator, device=device).to(x)
                        )
                        * strength ** i
                )
                if w == 1 or h == 1:
                    break  # Lowest resolution is 1x1
        else:
            raise ValueError(f"unknown downscale strategy: {downscale_strategy}")

        noise = noise / noise.std()  # Scaled back to roughly unit variance
        return noise

    @torch.no_grad()
    def __call__(
            self,
            input_image: Union[Image.Image, torch.Tensor],
            denoising_steps: Optional[int] = None,
            processing_res: Optional[int] = None,
            resample_method: str = "bilinear",
            generator: Union[torch.Generator, None] = None,
            prompt: str = None,
    ) -> DenoiseImgOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
        Returns:
            `DenoiseImgOutput`: Output class for prediction pipeline, including:
            - **gt_np** (`np.ndarray`) Predicted gt map, with values in the range of [0, 1]
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
                4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Predict depth maps
        gt_pred_raw = self.single_infer(
            rgb_in=rgb_norm,
            num_inference_steps=denoising_steps,
            generator=generator,
        )

        gt_preds = gt_pred_raw.detach()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # Convert to numpy
        gt_pred = gt_preds.squeeze()

        # Clip output range
        assert gt_pred.min() >= -1.0 and gt_pred.max() <= 1.0

        return DenoiseImgOutput(
            gt_np=gt_pred,
        )

    @torch.no_grad()
    def single_infer(
            self,
            rgb_in: torch.Tensor,
            num_inference_steps: int,
            generator: Union[torch.Generator, None] =None,
            prompt: str = None,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`): range [-1,1]
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.diff.text_encoder.device

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.diff.norm_img_to_latent(rgb_in)

        # Initial depth map (noise)
        gt_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        if prompt == None:
            # Batched empty text embedding
            if self.empty_text_embed is None:
                self.encode_empty_text()
            text_embed = self.empty_text_embed.repeat(
                (rgb_latent.shape[0], 1, 1)
            ).to(device)  # [B, 2, 1024]
        else:
            negative_text_embed, text_embed = self.get_text_embedding_(prompt, self.diff.text_encoder.device)
            text_embed = torch.cat([negative_text_embed, text_embed])

        # Denoising loop
        iterable = enumerate(timesteps)
        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, gt_latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.predict_(unet_input, t, text_embed, prompt)

            # compute the previous noisy sample x_t -> x_t-1
            gt_latent = self.scheduler.step(
                noise_pred, t, gt_latent, generator=generator
            ).prev_sample

        gt = self.diff.latent_to_img_tensor(gt_latent)
        return gt


    def predict_(self, unet_input, t, text_embed, prompt):
        if prompt == None:
            noise_pred = self.diff.unet(
                unet_input, t, encoder_hidden_states=text_embed
            ).sample  # [B, 4, h, w]
        else:
            if self.low_resources == False:
                noise_pred = self.diff.unet(
                    torch.cat([unet_input] * 2), t,
                    encoder_hidden_states=text_embed
                ).sample  # [B, 4, h, w]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                negative_text_embed, text_embed = text_embed.chunk(2)
                noise_pred_uncond = self.diff.unet(
                    unet_input, t, encoder_hidden_states=negative_text_embed
                ).sample  # [B, 4, h, w]
                noise_pred_text = self.diff.unet(
                    unet_input, t, encoder_hidden_states=text_embed
                ).sample  # [B, 4, h, w]
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred


    def get_text_embedding_(self, prompt, device, negative_prompt=None,
                            do_classifier_free_guidance=True, num_images_per_prompt=1):
        prompt_embeds, negative_prompt_embeds = self.diff.pipe.encode_prompt(
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
        text_inputs = self.diff.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.diff.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.diff.text_encoder.device)
        self.empty_text_embed = self.diff.text_encoder(text_input_ids)[0].to(self.dtype)

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, ):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")
