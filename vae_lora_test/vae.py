# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence, List, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn

from diffusers.utils import BaseOutput, is_torch_version
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention_processor import SpatialNorm
from .unet_2d_blocks_v2 import UNetMidBlock2D, get_down_block, get_up_block



@dataclass
class DecoderOutput(BaseOutput):
    """
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Decoded output sample of the model. Output of the last layer of the model.
    """

    sample: torch.FloatTensor


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
        attention_processor : Optional[ str ] = 'lora',
        mid_block_add_attention : Optional[ bool ] = True, # 用於配合 v0.29.2 的呼叫機制，本身無用處
    ):
        super().__init__()
        
        """
        Arguments for Decoder in stable diffusion v1.5
        in_channels: 3
        out_channels: 4
        up_block_type: [ 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D' ]
        block_out_channels: [128, 256, 512, 512 ]
        layers_per_block: 2
        norm_num_groups: 32
        act_fn: 'silu'
        double_z: True
        attention_processor : None
        """
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        
        # down
        # down_block_types 是連續四組 `DownEncoderBlock2D`
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            attention_processor = attention_processor,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def set_gradient_flag( self ) -> None:
        self.eval()
        self.mid_block.set_gradient_flag()
        

    def forward(
        self, 
        x           : torch.Tensor, 
        lora_scale  : float = 0.0
        ) -> torch.Tensor:
        
        sample : torch.Tensor = x
        sample : torch.Tensor = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                # middle
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)
                # sample 的縮小順序：
                # 1x128x512x512 -> 1x128x256x256
                # 1x128x256x256 -> 1x256x128x128
                # 1x256x128x128 -> 1x512x64x64
                # 1x512x64x64   -> 1x512x64x64

            # middle
            sample = self.mid_block( sample, scale = lora_scale )

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        norm_type="group",  # group, spatial
        attention_processor : Optional[ str ] = 'lora',
        mid_block_add_attention : Optional[ bool ] = True, # 用於配合 v0.29.2 的呼叫機制，本身無用處
    ):
        """
        Arguments for Decoder in stable diffusion v1.5
        Stable Diffusion v1.5 預設模式下各參數實際內容如下：

        1. in_channels: 4
        2. out_channels: 3
        3. up_block_type: [ 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D' ]
        4. block_out_channels: [128, 256, 512, 512 ]
        5. layers_per_block: 2
        6. norm_num_groups: 32
        7. act_fn: 'silu'
        8. norm_type: 'group'
        
        """
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            attention_processor = attention_processor
        )

        # up
        # up_block_types: UpDecoderBlock2D x4
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(
        self, 
        z : torch.Tensor, 
        lora_scale : float = 0.0,
        latent_embeds : torch.Tensor = None, # 這個不用
        latent_conds : Optional[ Dict[ str, List[ torch.Tensor ] ] ] = None,
        scales_conds : Optional[ List[ float ] ] = None,
        flag_control : Optional[ bool ] = False, 
        ) -> Union[ torch.Tensor, Tuple[ torch.Tensor, List[ torch.Tensor ] ] ]:
        """
        Decoder.forward:

        Args:
        ----------
        z: torch.Tensor
        來自 encoder/unet 的 latent embedding
        ----------
        latent_embeds: torch.Tensor
        原始 AutoEncoderKL 中的設定，應是 time_embedding，無視即可
        ----------
        latent_conds: Optional[ Dict[ str, List[ torch.Tensor ] ] ]
        來自旁支 Decoder 中各個 block 的 embedding
        ----------
        scales_conds: Optional[ List[ float ] ]
        調和各個 latent_conds 提供的 embedding 影響力大小
        ----------
        flag_control: Optional[ bool ]
        決定不同的輸出種類：
        => True: Condition Decoder
        => False: Original Decoder

        Return:
        ----------
        Union[ torch.Tensor, Tuple[ torch.Tensor, List[ torch.Tensor ] ] ]
        
        case 1: torch.Tensor
        作為主幹的 Decoder 僅需輸出一張解碼樣本

        case 2: Tuple[ torch.Tensor, List[ torch.Tensor ] ]
        作為旁支的 Condition Decoder ，除了最後一層輸出外，
        需要輸出一組 embedding ，用於穿插在主幹的各個 up block 前後

        """

        # 檢查目前 Decoder 屬於提供 condition guidance 的旁枝，還是解碼 latent 的主支
        # flag_weighted: 是否有傳入 condition guidance
        if latent_conds is not None and scales_conds is not None:
            flag_weighted = True
        else:
            flag_weighted = False

        def weighted_embedding(
            latent      : torch.Tensor,  
            conds       : Dict[ str, List[ torch.Tensor ] ], 
            scales      : List[ float ],
            block_idx   : int,
            ) -> torch.Tensor:
            """
            weighted_embedding:
            將主幹的 latent 與旁支的 condition embeddings 以特定權重組合進行疊加

            Args:
            ----------
            latent: torch.Tensor
            主幹 embedding
            ----------
            conds: Dict[ str, List[ torch.Tensor ] ]
            旁支 condition embeddings 
            直接傳入 latent_conds 即可
            ----------
            scales: List[ float ]
            各個旁支 condition embedding 對應的 scale
            ----------
            block_idx: int
            標示當前 Decoder 位置的 index
            不同 Block 的輸入尺寸不同

            Return:
            ----------
            加總後的主幹 latent ，或是完全沒加總的 latent 

            """
            if flag_weighted is False:
                return latent
            
            for idx, name in enumerate( conds ):
                cond = conds[ name ] # 由 condition mode 存取對應的 embedding list
                embed = cond[ block_idx ] # 由 block_idx 挑出對應位置的 embedding
                embed = embed.to( device = latent.device, dtype = latent.dtype )
                if latent.shape != embed.shape: # 若 shape 不匹配就會報錯
                    raise ValueError( 
                        "vae, Decoder: Invalid condition embedding shape with latent {} and embed {}".format( 
                            latent.shape, embed.shape ) )
                latent = latent + embed * scales[ idx ] # 將 embedding 以對應權重與 latent 相加

            return latent
        
        # sample: 待解碼 latent
        sample = z
        sample : torch.Tensor = self.conv_in(sample)
        # latent_buffer: 用於儲存各個 Block 的輸出作為其它 Decoder 的 Condition embedding
        latent_buffer : List[ torch.Tensor ] = []

        upscale_dtype = next( iter( self.up_blocks.parameters() ) ).dtype
        # 這裡透過 self.up_blocks.parameters() 獲得所有 up_block 的參數
        # 由 iter() 將這個 self.up_blocks 變為 iterator
        # 由 next() 開始遞迴這個 iterator
        # 用途：在下方 if-else 中 sample.to() 將 sample 轉到與 up_block 相容的資料型態
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                """
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds, use_reentrant=False
                )
                """
                sample = self.mid_block( sample, scale = lora_scale )
                sample = sample.to(upscale_dtype)
                
                # 儲存 middle block 的輸出
                sample_clone = sample.clone().to( 'cpu' )
                latent_buffer.append( sample_clone ) # offload from gpu

                # up
                for level, up_block in enumerate( self.up_blocks ):
                    # 若存在 conditions guidance 就進行加權
                    sample = weighted_embedding( latent = sample, conds = latent_conds, scales = scales_conds, block_idx = level )

                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block), sample, latent_embeds, use_reentrant=False
                    )

                    # 儲存 up block 的輸出
                    sample_clone = sample.clone().to( 'cpu' )
                    latent_buffer.append( sample_clone ) # offload from gpu
            else:
                # middle
                """
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                """
                sample = self.mid_block( sample, scale = lora_scale )
                sample = sample.to(upscale_dtype)

                # 儲存 middle block 的輸出
                sample_clone = sample.clone().to( 'cpu' )
                latent_buffer.append( sample_clone )

                # up
                for level, up_block in enumerate( self.up_blocks ):
                    # 若存在 conditions guidance 就進行加權
                    sample = weighted_embedding( latent = sample, conds = latent_conds, scales = scales_conds, block_idx = level )

                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
                    
                    # 儲存 up block 的輸出
                    sample_clone = sample.clone().to( 'cpu' )
                    latent_buffer.append( sample_clone )
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds, scale = lora_scale )
            sample = sample.to(upscale_dtype)

            # 儲存 middle block 的輸出
            sample_clone = sample.clone().to( 'cpu' )
            latent_buffer.append( sample_clone )
            
            # up
            for level, up_block in enumerate( self.up_blocks ):
                # 若存在 conditions guidance 就進行加權
                sample = weighted_embedding( latent = sample, conds = latent_conds, scales = scales_conds, block_idx = level )

                sample = up_block(sample, latent_embeds)

                # 儲存 up block 的輸出
                sample_clone = sample.clone().to( 'cpu' )
                latent_buffer.append( sample_clone )

        # 若存在 condition guidance 就進行加權
        # 這裡 block_idx 用 -1 表示最後一層 condition embedding 
        sample = weighted_embedding( latent = sample, conds = latent_conds, scales = scales_conds, block_idx = -1 )
        
        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 由 flag_control 控制是否需要回傳 latent_buffer
        # 換句話說，就是決定這個 Decoder 是當作主幹還是 control
        if flag_control is True:
            return sample, latent_buffer
        else:
            return sample
    
class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self, n_e, vq_embed_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True
    ):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        min_encoding_indices = torch.argmin(torch.cdist(z_flattened, self.embedding.weight), dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape, generator=generator, device=self.parameters.device, dtype=self.parameters.dtype
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean
