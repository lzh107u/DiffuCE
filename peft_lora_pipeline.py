import os

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel, LoraConfig

from custom_backward import StableDiffusionImg2ImgModifiedPipeline
# from diffusers import StableDiffusionImg2ImgPipeline

from vae_lora_test import encode_latent
from glob import glob
from PIL import Image
from math import floor
from typing import Union, Optional, Tuple, Any

def get_lora_sd_pipeline(
    ckpt_dir                : str, 
    base_model_name_or_path : Optional[ str ] = None, 
    dtype                   : Optional[ torch.dtype ] = torch.float16, 
    device                  : Optional[ torch.device ] = "cuda", 
    adapter_name            : Optional[ str ] = "default"
):
    """
    
    
    """
    unet_sub_dir = os.path.join( ckpt_dir, "unet" )
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_name_or_path, torch_dtype=dtype, requires_safety_checker=False
    ).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe

def load_adapter(
    pipe        : StableDiffusionPipeline, 
    ckpt_dir    : str, 
    adapter_name: str ):

    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    pipe.unet.load_adapter(unet_sub_dir, adapter_name=adapter_name)
    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder.load_adapter(text_encoder_sub_dir, adapter_name=adapter_name)

def set_adapter(pipe, adapter_name):
    pipe.unet.set_adapter(adapter_name)
    if isinstance(pipe.text_encoder, PeftModel):
        pipe.text_encoder.set_adapter(adapter_name)


def merging_unet_with_peft( 
        unet            : UNet2DConditionModel, 
        ckpt_dir        : str, 
        adapter_name    : Optional[ str ] = "default" ) -> UNet2DConditionModel:
    """
    merging_unet_with_peft:
    將給予的 UNet( 預設為 SD V1.5 ) 與 peft lora 合併

    Args:
    ----------
    unet: UNet2DConditionModel
    待合併的 UNet
    ----------
    ckpt_dir: str
    peft lora 位置
    ----------
    adapter_name: str
    adapter 名字，預設為 `default`

    Return:
    ----------
    UNet2DConditionModel:
    與 peft lora 合併的 UNet

    """
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    if isinstance( unet, PeftModel ):
        unet.set_adapter( adapter_name )
    else:
        unet = PeftModel.from_pretrained( unet, unet_sub_dir, adapter_name = adapter_name )
    unet = unet.merge_and_unload()
    return unet

def merging_lora_with_text_enc( 
    text_encoder    : Any, 
    ckpt_dir        : str, 
    adapter_name    : Optional[ str ] = "default" ):
    """
    merging_lora_with_text_enc:
    將給予的 SD text encoder 與 peft lora 合併
    大都為 CLIP 或 Roberta-based 

    Args:
    ----------
    text_encoder: Any
    待合併的 text encoder
    ----------
    ckpt_dir: str
    peft lora 位置
    ----------
    adapter_name: str
    adapter 名字，預設為 `default`

    Return:
    ----------
    Any
    與 peft lora 合併後的 text encoder

    """
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir):
        if isinstance( text_encoder, PeftModel ):
            text_encoder.set_adapter(adapter_name)
        else:
            text_encoder = PeftModel.from_pretrained(
                text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
            )
        text_encoder = text_encoder.merge_and_unload()

    return text_encoder

def merging_lora_with_base(pipe, ckpt_dir, adapter_name = "default" ) -> StableDiffusionPipeline:
    """
    merging_lora_with_base:
    將 peft lora unit 的權重合併進原生 Stable Diffusion 內
    分為 UNet 與 text_encoder

    Args:
    ----------
    pipe: StableDiffusionPipeline
    待合併的權重
    ----------
    ckpt_dir: str
    儲存有 UNet 與 text_encoder 權重的資料夾
    ----------
    adapter_name: str
    愈合併的 adapter 的名字，預設為 `default`

    Return:
    ----------
    StableDiffusionPipeline
    合併後的 SD pipeline
    
    """
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if isinstance(pipe.unet, PeftModel):
        pipe.unet.set_adapter(adapter_name)
    else:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)
    pipe.unet = pipe.unet.merge_and_unload()
    print( 'peft-lora-pipeline, merging-lora-with-base: load unet' )

    if os.path.exists(text_encoder_sub_dir):
        if isinstance(pipe.text_encoder, PeftModel):
            pipe.text_encoder.set_adapter(adapter_name)
        else:
            pipe.text_encoder = PeftModel.from_pretrained(
                pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
            )
        pipe.text_encoder = pipe.text_encoder.merge_and_unload()
        print( 'peft-lora-pipeline, merging-lora-with-base: load text encoder' )

    return pipe


def create_weighted_lora_adapter( 
    pipe : StableDiffusionPipeline, 
    adapters, 
    weights, 
    adapter_name = "default" ) -> StableDiffusionPipeline:

    pipe.unet.add_weighted_adapter(adapters, weights, adapter_name)
    if isinstance(pipe.text_encoder, PeftModel):
        pipe.text_encoder.add_weighted_adapter(adapters, weights, adapter_name)

    return pipe

def pipeline( 
    ckpt_dir            : str,
    base_model          : str,
    device              : Optional[ torch.device ] = 'cuda',
    adapter_name        : Optional[ str ] = 'ct',
    dtype               : Optional[ torch.dtype ] = torch.float32,
    strength            : Optional[ float ] = 0.5,
    random_seed         : Optional[ int ] = 0,
    num_inference_steps : Optional[ int ] = 40,
    early_stop_step     : Optional[ int ] = 0 ) -> None:

    torch.manual_seed( random_seed )
    # 宣告一組 pipeline
    pipe = StableDiffusionImg2ImgModifiedPipeline.from_pretrained( pretrained_model_name_or_path = base_model )
    pipe = merging_lora_with_base( 
        pipe = pipe,
        ckpt_dir = ckpt_dir,
        adapter_name = adapter_name )
    pipe = pipe.to( device )
    ct_image_paths = glob( '{}/*.png'.format( CBCT_IMAGE_FOLDER ) )
    
    filtered_ct = Image.open( ct_image_paths[ 200 ] ).resize( ( 512, 512 ) ).convert( "RGB" )
    print( 'peft_lora_pipeline: load file: {}'.format( ct_image_paths[ 200 ] ) )


    # timestep, arg_encoder_dict, modified_latent 都註解掉
    timestep : int = floor( 1000 * strength ) + 1

    arg_encoder_dict = {
        "image" : filtered_ct,
        "timestep" : timestep,
        "dtype" : dtype,
        "flag_wavelet" : True,
    }
    modified_latent = encode_latent( **arg_encoder_dict )

    # 決定文字 prompt
    prompt = 'A clean CT image'
    generator = torch.Generator( device = device ).manual_seed( 0 )

    arg_pipe_dict = {
        "prompt" : prompt,
        "image" : filtered_ct,
        "strength" : strength,
        "num_inference_steps" : num_inference_steps,
        "generator" : generator,
        "modified_latent" : modified_latent,
        "early_stop_step" : early_stop_step,
    }
    # 把 modified_latent 註解掉
    out = pipe( **arg_pipe_dict )
    
    result : Image.Image = out[ 0 ][ 0 ]
    result.convert( "L" ).save( "your_peft_lora_output.png" )

    return 

def random_check():
    torch.manual_seed( 42 )

    sample = torch.randn( ( 2, 2 ) )
    print( sample )


if __name__ == '__main__':
    ckpt_dir = 'generated_lora'
    base_model = "runwayml/stable-diffusion-v1-5"
    adapter_name = 'ct'
    dtype = torch.float32
    CT_IMAGE_FOLDER = '0821_tuning/CT'
    CBCT_IMAGE_FOLDER = '0821_tuning/CBCT'

    if torch.cuda.is_available() is True:
        device = 'cuda'
    else:
        device = 'cpu'
    print( 'peft_lora_pipeline, device: {}'.format( device ) )

    param_pipeline = {
        'ckpt_dir' : ckpt_dir,
        'base_model' : base_model,
        'device' : device,
        'adapter_name' : adapter_name,
        'strength' : 0.4,
        'dtype' : dtype,
        'num_inference_steps' : 50,
        'early_stop_step' : 50
    }
    pipeline( **param_pipeline )
    
