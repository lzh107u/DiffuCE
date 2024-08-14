from PIL import Image
from .autoencoder_kl_v2 import AutoencoderKL
import diffusers
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers import (
    DDPMScheduler,
)
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable, Any, Union, Sequence, Optional
from pywt import dwt2
import numpy as np
from re import search
from .vae import Encoder, DiagonalGaussianDistribution, Decoder
from diffusers.models.attention_processor import LoRAAttnProcessor


LORA_NAME = 'lora_unit_epoch1000_mixed-ts.pt'
SYNTHRAD_LORA = 'align_synthrad2023_lora_unit.pt'

# LORA_NAME = 'align_wavelet_epoch600_lora_unit.pt'
pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'

POSSIBLE_PATH_PREFIX = [
    '',
    '/home/diffuce/vae_lora_test/',
]
VALID_RET_MODE = [ 'inference', 'controlnet' ] 

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def valid_retmode_check( mode : str ) -> str:
    if mode not in VALID_RET_MODE:
        mode = 'inference'
    return mode

def name_is_main( func ):
    # 以這個 file 為 __main__ 時才執行這個裝飾器回傳的函數
    def wrapper( *args, **kwargs ):
        if __name__ == '__main__':
            return func( *args, **kwargs )
        else:
            return None
    return wrapper

@name_is_main
def print_msg( msg : str = '' ):
    print( msg )

#### 將輸入的 np.ndarray 正規化到 0 ~ 1 之間 ####
def normalize( array: np.ndarray ) -> np.ndarray:
    array = array - np.min( array )
    array = array / np.max( array )
    return array
#### 將輸入的 np.ndarray 以 np.uint8 格式正規化到 0-255 之間 ####
def to_uint8_255( arr: np.ndarray ) -> np.ndarray:
    arr = normalize( arr )
    arr = np.uint8( arr * 255 )
    return arr

def vae_init( 
    model_name: str = pretrained_model_name_or_path,
    ) -> diffusers.AutoencoderKL:
    
    vae : AutoencoderKL = AutoencoderKL.from_pretrained( model_name, subfolder = "vae", low_cpu_mem_usage = False )
    return vae
    for prefix in POSSIBLE_PATH_PREFIX:
        try:
            vae.load_state_dict( state_dict = torch.load( '{}{}'.format( prefix, LORA_NAME ) ) )
            print_msg( msg = 'backward-exp, vae_init: pretrained VAE with name: `{}` loaded.'.format( LORA_NAME ) )
            return vae
        except:
            pass
    
    raise FileNotFoundError("Can't find lora checkpoint.")

def single_wavelet_2d(
    image : Union[ np.ndarray, Image.Image ],
    wavelet : Optional[ str ] = 'bior2.6'
    ) -> Tuple[ np.ndarray, np.ndarray ]:
    """
    single_wavelet_2d:
    透過 PyWavelets 提供的 2 維離散小波轉換分解輸入的影像訊號

    Args:
    ----------
    image: Union[ np.ndarray, Image.Image ]
    輸入影像( 訊號 )
    ----------
    wavelet: Optional[ str ]
    小波類型，這裡的格式採用 PyWavelets 內的命名原則，說明如下：
    => bior 是 biorthogonal 的意思，是一種小波函數
        => 尚有其他不同的小波函數，如 'haar'
    => 2 代表濾波器長度
        => 大濾波器：較能保留細節，適合分解高頻成分
        => 小濾波器：較能平滑信號，適合分解低頻成分
    => 6 代表正交階數，意為"6階正交"，一般來說階數越高，重構效能愈好

    注意：小波類型、濾波器大小與階數都不是隨便搭配的，比方說不存在 bior2.5
    詳細清單請參考 pywt.wavelist()

    Return:
    ----------
    Tuple[ np.ndarray, np.ndarray ]
    輸入訊號的低頻成分與對角線高頻成分

    """
    coeffs = dwt2( image, wavelet )
    LL, ( LH, HL, HH ) = coeffs

    return LL, HH

def vae_dict_init(
    flag_enc_lora : bool = True,
    flag_dec_lora : bool = False,
    ckpt_mode : Optional[ str ] = 'default',    
    ) -> Dict[ str, nn.Module ]:
    """
    
    """
    arg_encoder = {
        "in_channels" : 3,
        "out_channels" : 4,
        "down_block_types" : [
            'DownEncoderBlock2D',
            'DownEncoderBlock2D',
            'DownEncoderBlock2D',
            'DownEncoderBlock2D',
        ],
        "block_out_channels" : [ 128, 256, 512, 512 ],
        "layers_per_block" : 2,
        "norm_num_groups" : 32,
        "act_fn" : 'silu',
        "double_z" : True,
        "attention_processor" : "lora" if flag_enc_lora else "not lora", 
        # 這裡可以選 `lora` 或其他( 不是 `lora` 就行 )
    }

    arg_decoder = {
        "in_channels" : 4,
        "out_channels" : 3,
        "up_block_types" : [
            'UpDecoderBlock2D',
            'UpDecoderBlock2D',
            'UpDecoderBlock2D',
            'UpDecoderBlock2D',
        ],
        "block_out_channels" : [ 128, 256, 512, 512 ],
        "layers_per_block" : 2,
        "norm_num_groups" : 32,
        "act_fn" : "silu",
        "norm_type" : 'group',
        "attention_processor" : "lora" if flag_dec_lora else "not lora", 
    }

    encoder = Encoder( **arg_encoder )
    decoder = Decoder( **arg_decoder )
    quant_conv = nn.Conv2d( 2*4, 2*4, 1 )
    post_quant_conv = nn.Conv2d( 4, 4, 1 )
    # 這裡選用 4 的原因為 latent_channels 
    # stable diffusion 的 unet 是使用 4 通道的訊號
    # 詳細可參考 AutoEncoderKL 中 init 的相關定義

    if ckpt_mode not in [ 'default', 'synthrad' ]:
        ckpt_mode = 'default'
    if ckpt_mode == 'default' :
        lora_path = LORA_NAME
    elif ckpt_mode == 'synthrad':
        lora_path = SYNTHRAD_LORA

    for prefix in POSSIBLE_PATH_PREFIX:
        try:
            state_dict : Dict[ str, torch.Tensor ] = torch.load( '{}{}'.format( prefix, lora_path ) )
            print_msg( msg = 'backward-exp, vae_init: pretrained VAE with name: `{}` loaded.'.format( lora_path ) )
            break
        except:
            state_dict = None
            print_msg('not found in {}{}'.format( prefix, lora_path ))
            continue
    
    if state_dict is None:
        raise FileNotFoundError("Can't find lora checkpoint.")

    # state_dict : Dict[ str, torch.Tensor ] = torch.load( LORA_NAME )
    encoder_param = {}
    decoder_param = {}
    quant_param = {}
    post_quant_param = {}
    pattern_enc = r'encoder'
    pattern_dec = r'decoder'

    for idx, name in enumerate( state_dict ):
        # 注意：個別宣告時，第一個前綴要拿掉
        # 因為那是在更上一層，以 AutoEncoderKL 為主體一起宣告時才有的名字
        name_split = name.split( '.' )
        new_name = '.'.join( name_split[ 1: ] )
        # encoder
        result = search( pattern_enc, name )
        if result: 
            encoder_param[ new_name ] = state_dict[ name ]
            continue
        # decoder
        result = search( pattern_dec, name )
        if result:
            decoder_param[ new_name ] = state_dict[ name ]
            continue
        # quant_conv & post_quant_conv
        if 'quant_conv' in name_split:
            quant_param[ new_name ] = state_dict[ name ]
        if 'post_quant_conv' in name_split:
            post_quant_param[ new_name ] = state_dict[ name ]
    # load state_dict
    ret = encoder.load_state_dict( encoder_param, False )
    # print( 'ret from encoder: {}'.format( ret ) )
    ret = decoder.load_state_dict( decoder_param, False )
    # print( 'ret from decoder: {}'.format( ret ) )

    quant_conv.load_state_dict( quant_param )
    post_quant_conv.load_state_dict( post_quant_param )

    vae_dict = {
        "encoder" : encoder.eval(),
        "decoder" : decoder.eval(),
        "quant_conv" : quant_conv.eval(),
        "post_quant_conv" : post_quant_conv.eval()
    }

    return vae_dict

def preprocess(
    image : Image.Image,
    flag_wavelet : Optional[ bool ] = False,
    ) -> torch.Tensor:

    # 若開啟小波模式，則影像將先進行一次 2-level discrete 2D wavelet transform
    if flag_wavelet is True:
        # 注意：小波轉換僅允許單通道灰度影像通過
        image = image.convert( "L" )
        # 連續兩次低頻的小波轉換
        low, high = single_wavelet_2d( image = image )
        low, high = single_wavelet_2d( image = low )
        # 小波轉換的輸入並不限定是 np.ndarray
        # 但輸出肯定是 np.ndarray 且資料型態並不是 uint8 ，需要正規化
        low = to_uint8_255( arr = low )
        image : Image.Image = Image.fromarray( low ).convert( "RGB" ).resize( ( 512, 512 ) )
        # 小波輸入模式下要更動讀取的 vae
        LORA_NAME = 'align_wavelet_lora_unit_v2.pt'
    
    image_processor = VaeImageProcessor( vae_scale_factor = 8 )
    # 若輸入影像不是 torch.Tensor ，就需要 image_processor 協助轉換
    if not isinstance( image, torch.Tensor ):
        # image = image.convert( "RGB" ).resize( ( 512, 512 ) )
        image : torch.Tensor = image_processor.preprocess( image )
    return image

def scheduler_setup() -> DDPMScheduler:
    # 載入 noise scheduler ，用於對 timesteps 給予相對應的 noise 強度
    noise_scheduler = None
    for prefix in POSSIBLE_PATH_PREFIX:
        try:
            noise_scheduler : DDPMScheduler = DDPMScheduler.from_pretrained( pretrained_model_name_or_path = '{}scheduler_config.json'.format( prefix ) )
        except:
            pass
    if noise_scheduler is None:
        raise FileNotFoundError( "backward_experiment: Can't find scheduler_config.json." )
    return noise_scheduler

def encode_latent(
    image       : Union[ Image.Image, torch.Tensor ],
    timestep    : Union[ torch.Tensor, int ],
    dtype       : Optional[ torch.dtype ] = torch.float16,
    lora_scale  : Optional[ float ] = 1.0,
    random_seed : Optional[ int ] = 0,
    ret_mode    : str = 'inference',
    vae         : Optional[ AutoencoderKL ] = None,
    flag_noise  : Optional[ bool ] = True,
    flag_wavelet: Optional[ bool ] = False,
    ) -> Union[ torch.Tensor, Tuple[ torch.Tensor, torch.Tensor, torch.Tensor ] ]:
    """
    encode_latent:
    使用 Latent Alignment 實驗的 VAE encoder 將來自 CBCT 的 Latent 微調到 CT domain 上

    Args:
    ----------
    image: Image.Image
    要 encode 的原圖，這裡預設是已經在 dicom_utils 中進行過相關前處理
    ----------
    timestep: Union[ torch.Tensor, int ]
    latent 在整條 timesteps 軸線上的位置，這裡對應到原生 stable diffusion pipeline 中 prepare_timesteps 的那步驟
    原始 pipeline 中透過 strength 調整 backward process 開始的位置，這裡就是直接使用那個位置( e.g. 701, 601, ... )
    ----------
    dtype: torch.dtype 
    愈輸出的 latent dtype ，這裡需要與下游的 unet 與 decoder 相同
    ----------
    lora_scale: float
    VAE encoder 中 LoRA 組件的強度，從目視結果推測並沒有差多少，但無法確定在 latent space 中是否如此，暫時使用 1.0
    ----------
    random_seed: int
    固定住加躁功能的隨機種子，已透過實際實驗證明不同的採樣躁聲會讓 unet 輸出產生變化
    ----------
    flag_noise: bool
    是否需要加噪
    這裡用於不加噪且走極短步數的實驗
    ----------
    flag_wavelet: bool
    是否使用小波影像作為輸入

    Return:
    ----------
    torch.Tensor
    經過 vae 微調並編碼的 CBCT latent

    Tuple[ torch.Tensor x3 ]
    依序為：
    noisy_latent: 經過編碼、微調與加躁的 CBCT latent
    noise: 用於加躁的高斯雜訊
    init_latent: 尚未經過加躁的 latent
    
    """
    if ret_mode not in [ 'inference', 'controlnet' ]:
        ret_mode = 'inference'
    # 選定隨機種子，這裡將固定產生的 gaussian noise
    torch.manual_seed( random_seed )
    
    # 載入調整好的 lora vae encoder
    if vae is None:
        vae = vae_init()
    if not isinstance( vae, AutoencoderKL ):
        print( 'custom-vae-encode, invalid given vae.' )
    vae.to( device = device, dtype = dtype )
    vae.eval()
    image = image.convert("RGB").resize( ( 512, 512 ) )
    image = preprocess( image = image, flag_wavelet = False )
    noise_scheduler = scheduler_setup()
    # vae 實際作業區域
    with torch.no_grad():
        # 編碼與重參數化
        # print( "shape of image: {}".format( image.shape ) )
        init_latents = vae.encode( image.to( dtype = dtype, device = device ), lora_scale = lora_scale ).latent_dist.sample()
        
        # 調整影像值域( 類似 normalize )
        init_latents = vae.config.scaling_factor * init_latents
        init_latents = torch.cat( [ init_latents ], dim = 0 )
        # 採樣噪聲    
        noise = torch.randn_like( init_latents, device = device, dtype = torch.float16 )

        if isinstance( timestep, int ):
            timestep : torch.Tensor = torch.tensor( data = [ timestep ], device = device )

        # 依照給予的 timestep 由 noise scheduler 加上相應強度的 noise
        if flag_noise is True:
            noisy_latent : torch.Tensor = noise_scheduler.add_noise( init_latents, noise, timestep )
        else:
            noisy_latent : torch.Tensor = init_latents
    # 將 vae 從 gpu 上卸下
    vae.to( device = 'cpu' )

    # 依照給予的 ret_mode 回傳對應的 tuple
    ret_mode = valid_retmode_check( mode = ret_mode )

    if ret_mode == 'inference':
        return noisy_latent
    elif ret_mode == 'controlnet':
        return noisy_latent, noise, init_latents

def encode_dict(
    image           : Union[ Image.Image, torch.Tensor ],
    timestep        : Union[ torch.Tensor, int ] = 0,
    scaling_factor  : float = 0.18215,
    device          : torch.device = 'cuda',
    dtype           : Optional[ torch.dtype ] = torch.float32,
    lora_scale      : Optional[ float ] = 1.0,
    random_seed     : Optional[ int ] = 0,
    ret_mode        : Optional[ str ] = 'inference',
    vae_dict        : Optional[ Dict[ str, nn.Module ] ] = None,
    flag_noise      : Optional[ bool ] = True,
    flag_wavelet    : Optional[ bool ] = False,
    flag_reparam    : Optional[ bool ] = True,
    flag_moments    : Optional[ bool ] = False,
    ) -> Union[ torch.Tensor, Tuple[ torch.Tensor, torch.Tensor, torch.Tensor ] ]:
    """
    encode_dict:
    這裡透過 vae_dict 將原先 AutoEncoderKL 的 encode 表現直接替換掉

    Args:
    ----------
    image: Image.Image
    要 encode 的原圖，這裡預設是已經在 dicom_utils 中進行過相關前處理
    ----------
    timestep: Union[ torch.Tensor, int ]
    latent 在整條 timesteps 軸線上的位置，這裡對應到原生 stable diffusion pipeline 中 prepare_timesteps 的那步驟
    原始 pipeline 中透過 strength 調整 backward process 開始的位置，這裡就是直接使用那個位置( e.g. 701, 601, ... )
    ----------
    scaling_factor: float
    用於對 latent 進行縮放，套用於 encode 之後與 decode 之前
    Stable Diffusion V1.5 的預設為 0.18215
    ----------
    dtype: torch.dtype 
    愈輸出的 latent dtype ，這裡需要與下游的 unet 與 decoder 相同
    ----------
    lora_scale: float
    VAE encoder 中 LoRA 組件的強度，從目視結果推測並沒有差多少，但無法確定在 latent space 中是否如此，暫時使用 1.0

    ## 2023-09-27 實驗：
    將 LoRA 使用在未加噪影像上會產生相當明顯的藍色色調
    ----------
    random_seed: int
    固定住加躁功能的隨機種子，已透過實際實驗證明不同的採樣躁聲會讓 unet 輸出產生變化
    ----------
    ret_mode: str
    決定回傳樣式，有 `inference` 與 `controlnet` 可選擇
    ----------
    ret_dict: Dict[ str, nn.Module ]
    裝載各個 AutoEncoderKL components 的 dict
    若未傳入將再透過 vae_dict_init() 初始化一組
    ----------
    flag_noise: bool
    是否需要加噪
    這裡用於不加噪且走極短步數的實驗
    ----------
    flag_wavelet: bool
    是否使用小波影像作為輸入
    ----------
    flag_reparam: bool
    輸出前是否重參數化
    ----------
    flag_moments: bool
    輸入是否為 moment，
    若為 False ，表示輸入尚未經過 encoder 提取特徵
    需要先過 encoder 再進行重參數化( reparameterize )；
    若為 True ，可直接進入重參數化

    Return:
    ----------
    torch.Tensor
    經過 vae 微調並編碼的 CBCT latent

    Tuple[ torch.Tensor x3 ]
    依序為：
    noisy_latent: 經過編碼、微調與加躁的 CBCT latent
    noise: 用於加躁的高斯雜訊
    init_latent: 尚未經過加躁的 latent
    
    """
    # 1. 前處理：
    ## 確定 image 與 ret_mode 在定義內
    if image is None:
        raise ValueError( 'Image is None !!' )
    
    if ret_mode not in [ 'inference', 'controlnet' ]:
        ret_mode = 'inference'
    ## 選定隨機種子，這裡將固定產生的 gaussian noise
    torch.manual_seed( random_seed )
    

    # 2. 初始化 noise_scheduler, encoder 與 image preprocessing
    ## 若 image 不是 moments ，則需要經過 Encoder 提取特徵
    ## 若 image 已經是 moments ，跳過這區直接重參數化
    if flag_moments is False:
        # 輸入資料前處理
        # 若輸入資料不屬於 torch.Tensor
        if isinstance( image, Image.Image ):
            image = image.convert( "RGB" ).resize( ( 512, 512 ) )
            image : torch.Tensor = preprocess( image = image, flag_wavelet = flag_wavelet )
        image = image.to( device, dtype = dtype )

        noise_scheduler = scheduler_setup()

    # shape check
    # dimension 必須是 1 x 3 x 512 x 512
    if len( image.shape ) > 4:
        image = torch.squeeze( image, dim = 0 )

    ## 初始化 encoder ( weight dict )
    if vae_dict is None:
        vae_dict = vae_dict_init()
    
    def extract_network( name: str ) -> nn.Module:
        # 取出特定的 AutoEncoderKL 組件
        network = vae_dict[ name ]
        network.to( device, dtype = dtype )
        network.eval()
        return network
    
    ## 取得 Encoder 與 quant_conv ，以用於 encode 
    encoder = extract_network( name = "encoder" )
    quant_conv = extract_network( name = "quant_conv" )
    

    # 3. 實際 encode 流程
    with torch.no_grad():
        # 3.1 對影像進行 encode
        ## 若輸入非 moments ，才需要 3.1
        if flag_moments is False:
            h = encoder( image, lora_scale ) # encode 
            moments = quant_conv( h ) # quantization
        else:
            moments = image.to( dtype = dtype, device = device )
        # 這裡可以直接輸出 mean & logvar
        # 注意：若打算對 latent 加噪，則無法直接回傳 moments
        if flag_reparam is False and flag_noise is False:
            return moments

        # 3.2 重參數化
        posterior = DiagonalGaussianDistribution( moments ) # form a distribution
        latent = posterior.sample() # reparameterization
        
        
        # 3.3 調整 latent 的區間
        latent = latent * scaling_factor
        latent = torch.cat( [ latent ], dim = 0 )
    

    # 3.4 offload from gpu( `cuda` )
    encoder.to( 'cpu' )
    quant_conv.to( 'cpu' )

    # 4. 添加 Gaussian Noise
    ## 若需要採樣並融合噪聲，才進這區
    if flag_noise is True:
        # 採樣一組隨機噪聲
        noise = torch.randn_like( latent, device = device, dtype = dtype )
        # 制定 timestep tensor，需要保持 int32/int64
        if isinstance( timestep, int ):
            timestep : torch.Tensor = torch.tensor( data = [ timestep ], device = device, dtype = torch.int32 )
        # 透過給定的 timestep 與 noise 在 latent 上進行混和
        noisy_latent = noise_scheduler.add_noise( latent, noise, timestep )
    else: # 選擇 flag_noise -> False ，即不對 latent 加噪
        noisy_latent = latent.clone()
    
    if ret_mode == 'inference':
        return noisy_latent
    elif ret_mode == 'controlnet':
        return noisy_latent, noise, latent

def normal_preprocess(
    image : Image.Image,
    ) -> torch.Tensor:
    
    image_processor = VaeImageProcessor( vae_scale_factor = 8 )
    # 若輸入影像不是 torch.Tensor ，就需要 image_processor 協助轉換
    if not isinstance( image, torch.Tensor ):
        # image = image.convert( "RGB" ).resize( ( 512, 512 ) )
        image : torch.Tensor = image_processor.preprocess( image )
    return image

def normal_vae_init(
    flag_enc_lora : bool = True,
    flag_dec_lora : bool = False,    
    ) -> Dict[ str, nn.Module ]:

    arg_encoder = {
        "in_channels" : 3,
        "out_channels" : 4,
        "down_block_types" : [
            'DownEncoderBlock2D',
            'DownEncoderBlock2D',
            'DownEncoderBlock2D',
            'DownEncoderBlock2D',
        ],
        "block_out_channels" : [ 128, 256, 512, 512 ],
        "layers_per_block" : 2,
        "norm_num_groups" : 32,
        "act_fn" : 'silu',
        "double_z" : True,
        "attention_processor" : "lora" if flag_enc_lora else "not lora", 
        # 這裡可以選 `lora` 或其他( 不是 `lora` 就行 )
    }

    arg_decoder = {
        "in_channels" : 4,
        "out_channels" : 3,
        "up_block_types" : [
            'UpDecoderBlock2D',
            'UpDecoderBlock2D',
            'UpDecoderBlock2D',
            'UpDecoderBlock2D',
        ],
        "block_out_channels" : [ 128, 256, 512, 512 ],
        "layers_per_block" : 2,
        "norm_num_groups" : 32,
        "act_fn" : "silu",
        "norm_type" : 'group',
        "attention_processor" : "lora" if flag_dec_lora else "not lora", 
    }

    encoder = Encoder( **arg_encoder )
    decoder = Decoder( **arg_decoder )
    quant_conv = nn.Conv2d( 2*4, 2*4, 1 )
    post_quant_conv = nn.Conv2d( 4, 4, 1 )
    # 這裡選用 4 的原因為 latent_channels 
    # stable diffusion 的 unet 是使用 4 通道的訊號
    # 詳細可參考 AutoEncoderKL 中 init 的相關定義

    for prefix in POSSIBLE_PATH_PREFIX:
        try:
            state_dict : Dict[ str, torch.Tensor ] = torch.load( '{}{}'.format( prefix, LORA_NAME ) )
            print_msg( msg = 'backward-exp, vae_init: pretrained VAE with name: `{}` loaded.'.format( LORA_NAME ) )
            break
        except:
            print_msg('not found in {}{}'.format( prefix, LORA_NAME ))
            continue
    
        raise FileNotFoundError("Can't find lora checkpoint.")

    # state_dict : Dict[ str, torch.Tensor ] = torch.load( LORA_NAME )
    encoder_param = {}
    decoder_param = {}
    quant_param = {}
    post_quant_param = {}
    pattern_enc = r'encoder'
    pattern_dec = r'decoder'

    for idx, name in enumerate( state_dict ):
        # 注意：個別宣告時，第一個前綴要拿掉
        # 因為那是在更上一層，以 AutoEncoderKL 為主體一起宣告時才有的名字
        name_split = name.split( '.' )
        new_name = '.'.join( name_split[ 1: ] )
        # encoder
        result = search( pattern_enc, name )
        if result: 
            encoder_param[ new_name ] = state_dict[ name ]
            continue
        # decoder
        result = search( pattern_dec, name )
        if result:
            decoder_param[ new_name ] = state_dict[ name ]
            continue
        # quant_conv & post_quant_conv
        if 'quant_conv' in name_split:
            quant_param[ new_name ] = state_dict[ name ]
        if 'post_quant_conv' in name_split:
            post_quant_param[ new_name ] = state_dict[ name ]
    # load state_dict
    ret = encoder.load_state_dict( encoder_param, False )
    # print( 'ret from encoder: {}'.format( ret ) )
    ret = decoder.load_state_dict( decoder_param, False )
    # print( 'ret from decoder: {}'.format( ret ) )

    quant_conv.load_state_dict( quant_param )
    post_quant_conv.load_state_dict( post_quant_param )

    vae_dict = {
        "encoder" : encoder.eval(),
        "decoder" : decoder.eval(),
        "quant_conv" : quant_conv.eval(),
        "post_quant_conv" : post_quant_conv.eval()
    }

    return vae_dict

def normal_encode(
    image           : Union[ Image.Image, torch.Tensor ],
    scaling_factor  : float = 0.18215,
    device          : torch.device = 'cuda',
    dtype           : Optional[ torch.dtype ] = torch.float32,
    lora_scale      : Optional[ float ] = 1.0,
    random_seed     : Optional[ int ] = 0,
    ret_mode        : Optional[ str ] = 'inference',
    vae_dict        : Optional[ Dict[ str, nn.Module ] ] = None,
    flag_wavelet    : Optional[ bool ] = False,
    flag_reparam    : Optional[ bool ] = True,
    flag_moments    : Optional[ bool ] = False,
    ) -> Union[ torch.Tensor, Tuple[ torch.Tensor, torch.Tensor, torch.Tensor ] ]:
    """
    encode_dict:
    這裡透過 vae_dict 將原先 AutoEncoderKL 的 encode 表現直接替換掉

    Args:
    ----------
    image: Image.Image
    要 encode 的原圖，這裡預設是已經在 dicom_utils 中進行過相關前處理
    ----------
    timestep: Union[ torch.Tensor, int ]
    latent 在整條 timesteps 軸線上的位置，這裡對應到原生 stable diffusion pipeline 中 prepare_timesteps 的那步驟
    原始 pipeline 中透過 strength 調整 backward process 開始的位置，這裡就是直接使用那個位置( e.g. 701, 601, ... )
    ----------
    scaling_factor: float
    用於對 latent 進行縮放，套用於 encode 之後與 decode 之前
    Stable Diffusion V1.5 的預設為 0.18215
    ----------
    dtype: torch.dtype 
    愈輸出的 latent dtype ，這裡需要與下游的 unet 與 decoder 相同
    ----------
    lora_scale: float
    VAE encoder 中 LoRA 組件的強度，從目視結果推測並沒有差多少，但無法確定在 latent space 中是否如此，暫時使用 1.0

    ## 2023-09-27 實驗：
    將 LoRA 使用在未加噪影像上會產生相當明顯的藍色色調
    ----------
    random_seed: int
    固定住加躁功能的隨機種子，已透過實際實驗證明不同的採樣躁聲會讓 unet 輸出產生變化
    ----------
    ret_mode: str
    決定回傳樣式，有 `inference` 與 `controlnet` 可選擇
    ----------
    ret_dict: Dict[ str, nn.Module ]
    裝載各個 AutoEncoderKL components 的 dict
    若未傳入將再透過 vae_dict_init() 初始化一組
    ----------
    flag_noise: bool
    是否需要加噪
    這裡用於不加噪且走極短步數的實驗
    ----------
    flag_wavelet: bool
    是否使用小波影像作為輸入
    ----------
    flag_reparam: bool
    輸出前是否重參數化
    ----------
    flag_moments: bool
    輸入是否需要提取特徵，
    若為 False ，則可直接進行重參數化( reparameterize )

    Return:
    ----------
    torch.Tensor
    經過 vae 微調並編碼的 CBCT latent

    Tuple[ torch.Tensor x3 ]
    依序為：
    noisy_latent: 經過編碼、微調與加躁的 CBCT latent
    noise: 用於加躁的高斯雜訊
    init_latent: 尚未經過加躁的 latent
    
    """
    if image is None:
        raise ValueError( 'Image is None !!' )
    
    if ret_mode not in [ 'inference', 'controlnet' ]:
        ret_mode = 'inference'
    # 選定隨機種子，這裡將固定產生的 gaussian noise
    torch.manual_seed( random_seed )
    
    # 若 image 不是 moments ，則需要經過 Encoder 提取特徵
    if flag_moments is False:
        # 輸入資料前處理
        # 若輸入資料不屬於 torch.Tensor
        if isinstance( image, Image.Image ):
            image = image.convert( "RGB" ).resize( ( 512, 512 ) )
            image : torch.Tensor = normal_preprocess( image = image )
        image = image.to( device, dtype = dtype )

    if vae_dict is None:
        vae_dict = vae_dict_init()
    
    def extract_network( name: str ) -> nn.Module:
        # 取出特定的 AutoEncoderKL 組件
        network = vae_dict[ name ]
        network.to( device, dtype = dtype )
        network.eval()
        return network
    
    # 取得 Encoder 與 quant_conv ，以用於 encode 
    encoder = extract_network( name = "encoder" )
    quant_conv = extract_network( name = "quant_conv" )
    
    # 實際 encode 流程
    with torch.no_grad():
        # 若輸入非 moments ，則需要透過 Encoder 進行處理
        if flag_moments is False:
            h = encoder( image, lora_scale ) # encode 
            moments = quant_conv( h ) # quantization
        else:
            moments = image.to( dtype = dtype, device = device )
        # 這裡可以直接輸出 mean & logvar
        # 注意：若打算對 latent 加噪，則無法直接回傳 moments
        if flag_reparam is False:
            return moments

        posterior = DiagonalGaussianDistribution( moments ) # form a distribution
        latent = posterior.sample() # reparameterization
        
        # 調整 latent 的區間
        latent = latent * scaling_factor
        latent = torch.cat( [ latent ], dim = 0 )
    # offload from gpu( `cuda` )
    encoder.to( 'cpu' )
    quant_conv.to( 'cpu' )
    
    return latent
    

def decode_latent(
    embedding   : torch.Tensor,
    dtype       : torch.dtype = torch.float16, 
    vae         : Optional[ AutoencoderKL ] = None,
    ) -> Image.Image:
    """
    decode_latent: 
    輸入 unet 算出的 embedding 或直接來自 vae encoder 的 embedding
    將其轉換成 image domain 上的樣式，並提供資料型態轉換

    Args:
    ----------
    embedding: torch.Tensor
    待解碼的 embedding
    ----------
    dtype: torch.dtype
    資料型態，決定了 decoder 與 embedding 的實際資料型態
    ----------
    vae: AutoencoderKL
    解碼器，這裡使用 stable diffusion v1.5 的預設權重
    目前的實驗都沒有更動 latent domain 位置，故可假設解碼器仍可起到效用

    Return:
    ----------
    Image.Image
    解碼後的影像

    """
    if vae is None or not isinstance( vae, AutoencoderKL ) or not isinstance( vae, diffusers.AutoencoderKL ):
        vae = vae_init()
    vae.to( device = device, dtype = dtype )
    vae.eval()
    out : torch.Tensor = vae.decode( embedding.to( dtype = dtype, device = device ) / vae.config.scaling_factor, return_dict = False )[ 0 ]
    out = out.cpu().detach().to( dtype = torch.float32 )
    
    image_processor = VaeImageProcessor( vae_scale_factor = 8 )

    result : List[ Image.Image ] = image_processor.postprocess( image = out )
    
    return result[ 0 ]

def decode_latent_with_dict(
    embedding           : torch.Tensor,
    dtype               : torch.dtype = torch.float16, 
    vae_dict            : Optional[ Dict[ str, nn.Module ] ] = None,
    vae_scale_factor    : Optional[ int ] = 8,
    scaling_factor      : float = 0.18215,
    device              : torch.device = 'cuda',
    flag_postprocess    : Optional[ bool ] = True
    ) -> Union[ Image.Image, torch.Tensor ]:
    """
    decode_latent_with_dict:
    透過 vae_dict 內各 components 完成 decoder 的工作
    這裡會直接將來自 Encoder 或 unet 的輸出直接轉換為 Image.Image
    無需其它後處理

    Args:
    ----------
    embedding: torch.Tensor
    待解碼的 latent
    ----------
    dtype: torch.dtype
    變數型態，一般來說在 torch.float16 / float32 間選擇
    ----------
    vae_dict: Dict[ str, nn.Module ]
    存有原 AutoEncoderKL 內各 component 的 dictionary
    ----------
    vae_scale_factor: int
    AutoEncoderKL 的縮放比，這裡採 Stable Diffusion v1.5 的預設
    ----------
    scaling_factor: float
    用於對 latent 進行縮放，在 encode 後與 decode 前皆需要這個動作
    Stable Diffusion v1.5 的預設值為 0.18215
    ----------
    device: torch.device
    latent 與 components 要執行的位置，建議選用 `cuda`

    Return:
    ----------
    Image.Image
    解碼且後處理過的影像

    """
    if vae_dict is None:
        vae_dict = vae_dict_init()

    image_processor = VaeImageProcessor( vae_scale_factor = vae_scale_factor )

    post_quant_conv = vae_dict[ "post_quant_conv" ]
    decoder = vae_dict[ "decoder" ]
    embedding = embedding / scaling_factor
    
    post_quant_conv.to( device, dtype = dtype )
    post_quant_conv.eval()
    decoder.to( device, dtype = dtype )
    decoder.eval()
    with torch.no_grad():
        z = post_quant_conv( embedding.to( dtype = dtype, device = device ) )
        dec : torch.Tensor = decoder( z )
        dec = dec.cpu().detach().to( dtype = torch.float32 )
    decoder.to( device = 'cpu' )
    
    if flag_postprocess is True:
        result : List[ Image.Image ] = image_processor.postprocess( dec, do_denormalize = [ True ] )
    else:
        return dec
    
    return result[ 0 ]

def pipeline_validation():
    image = Image.open( '200_cbct.png' ).resize( ( 512, 512 ) ).convert( "RGB" )
    vae_dict = vae_dict_init()
    arg_encoder = {
        "image" : image,
        "flag_noise" : False,
        "flag_wavelet" : False,
        "lora_scale" : 0.0,
        "vae_dict" : vae_dict,
    }
    emb = encode_dict( **arg_encoder )
    # result = decode_latent( embedding = emb )
    result = decode_latent_with_dict( embedding = emb, vae_dict = vae_dict )
    result.convert( 'RGB' ).save( 'new_decoder.png' )



def weight_validation():
    state_dict : Dict[ str, torch.Tensor ] = torch.load( LORA_NAME )
    
    count = 0
    pattern_lora = r'lora'
    pattern_enc = r'encoder'
    pattern_dec = r'decoder'
    encoder_param = {}
    decoder_param = {}

    for idx, name in enumerate( state_dict ):
        result = search( pattern_enc, name )
        if result:
            encoder_param[ name ] = state_dict[ name ]
            continue

        result = search( pattern_dec, name )
        if result:
            decoder_param[ name ] = state_dict[ name ]
            continue
        
        print( 'unknown params: {}'.format( name ) )
    
    arg_encoder = {
        "in_channels" : 3,
        "out_channels" : 4,
        "up_block_type" : [
            'DownEncoderBlock2D',
            'DownEncoderBlock2D',
            'DownEncoderBlock2D',
            'DownEncoderBlock2D',
        ],
        "block_out_channels" : [ 128, 256, 512, 512 ],
        "layers_per_block" : 2,
        "norm_num_groups" : 32,
        "act_fn" : 'silu',
        "double_z" : True,
        "attention_processor" : None,
    }

    arg_decoder = {
        "in_channels" : 4,
        "out_channels" : 3,
        "up_block_type" : [
            'UpDecoderBlock2D',
            'UpDecoderBlock2D',
            'UpDecoderBlock2D',
            'UpDecoderBlock2D',
        ],
        "block_out_channels" : [ 128, 256, 512, 512 ],
        "layers_per_block" : 2,
        "norm_num_groups" : 32,
        "act_fn" : "silu",
        "norm_type" : 'group',
    }

    return 

class DBE :
    """
    DBE( Domain Bridging Encoder ):
    
    DiffuCE 項目中的 Encoder ，基底為 HuggingFace Stable Diffusion V1.5 的 Encoder 。
    用於將影像從 Pixel domain 上轉換至 Latent Space 中，
    這裡額外使用 LoRA 將 CBCT 影像在 Latent Space 中的位置與 CT 靠近，
    具體實作方式請參閱 `DiffuCE: Expert-Level CBCT Image Enhancement using a Novel Conditional Denoising Diffusion Model with Latent Alignment`
    """
    def __init__( 
        self,
        flag_enc_lora : bool = True,
        flag_dec_lora : bool = False,
        ckpt_mode : Optional[ str ] = 'default',
        ) -> None:
        
        self.flag_enc_lora = flag_enc_lora
        self.flag_dec_lora = flag_dec_lora

        if ckpt_mode not in [ 'default', 'synthrad' ]:
            ckpt_mode = 'default'

        self.ckpt_mode = ckpt_mode

        self.encoder_dict = None
        self.set_encoder() # 取得 self.encoder_dict

        pass

    def set_encoder( 
        self,
        flag_enc_lora : bool = True,
        flag_dec_lora : bool = False, 
        ) -> None:
        """
        set_encoder:
        重設 DBE 的 encoder_dict ，透過呼叫 vae_dict_init() 進行更新
        """

        if isinstance( flag_enc_lora, bool ):
            self.flag_enc_lora = flag_enc_lora
        if isinstance( flag_dec_lora, bool ):
            self.flag_dec_lora = flag_dec_lora
        
        self.encoder_dict = vae_dict_init( 
            flag_enc_lora = self.flag_enc_lora, 
            flag_dec_lora = self.flag_dec_lora, 
            ckpt_mode = self.ckpt_mode )

    def encode(
        self,
        image           : Union[ Image.Image, torch.Tensor ],
        mode            : str = None,
        timestep        : int = 0,
        random_seed     : Optional[ int ] = 0,
        dtype           : torch.dtype = torch.float32,

        lora_scale      : float = 0.0,
        flag_noise      : Optional[ bool ] = True,
        flag_wavelet    : Optional[ bool ] = False,
        flag_reparam    : Optional[ bool ] = True,
        ) -> torch.Tensor:
        # self.encoder_dict 必須先初始化
        if self.encoder_dict is None:
            print( 'DBE.encode: `encoder_dict` has to be initialized first.' )
            return image
        # 可透過 mode 快速設定部分參數
        if mode is not None and mode in [ 'main', 'cond', 'ablation' ]:
            if mode == 'main':
                flag_noise = True
                flag_wavelet = False
                flag_reparam = True
                lora_scale = 1.0
            elif mode == 'cond':
                flag_noise = False
                flag_wavelet = False
                flag_reparam = True
                lora_scale = 0.0
            elif mode == 'ablation':
                flag_noise = True
                flag_wavelet = False
                flag_reparam = True
                lora_scale = 0.0
        
        # 若 encoder 沒有使用 lora layer ，則 lora_scale 寫死為 0.0
        if self.flag_enc_lora is False:
            lora_scale = 0.0
                
        # 整理參數並呼叫 encode_dict()
        arg_encode_dict = {
            "image" : image,
            "timestep" : timestep,
            "random_seed" : random_seed,
            "dtype" : dtype,
            "lora_scale" : lora_scale,
            "flag_noise" : flag_noise,
            "flag_wavelet" : flag_wavelet,
            "flag_reparam" : flag_reparam,
            "vae_dict" : self.encoder_dict,
        }
        # 送進去 encode ，device 會由 encode_dict() 負責處理
        latent = encode_dict( **arg_encode_dict )
        return latent



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    pipeline_validation()
    sample = torch.randn( size = ( 1, 3, 512, 512 ) )
    # weight_validation()
    # vae_dict_init()
    