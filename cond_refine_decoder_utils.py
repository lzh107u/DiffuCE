import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import ( 
    Dict, List, Tuple,
    Any, Union )
from tqdm import tqdm

from diffusers.image_processor import VaeImageProcessor

from vae_lora_test import (
    Decoder
)

from re import search
from glob import glob

POSSIBLE_PATH_PREFIX = [
    'vae_lora_test/',
    '/home/zihong/CBCT/vae_lora_test/',
]

LATENT_ROOT = '0915_tuning/latent'
LORA_NAME = 'lora_unit_epoch1000_mixed-ts.pt'

PRETRAINED_FOLDER = 'pretrained_weights/decoder/decoder-004-main-lpips'
DECODER_LORA = None
DECODER_AIR_LORA = None
DECODER_BONE_LORA = None

def _version_set(
    version : int = 5,
    final   : bool = True,
    ) -> None:

    global PRETRAINED_FOLDER
    global DECODER_LORA
    global DECODER_AIR_LORA
    global DECODER_BONE_LORA

    if final is True:
        sub_version = 'final'
    else:
        sub_version = 'best'

    if version == 5:
        PRETRAINED_FOLDER = 'pretrained_weights/decoder/decoder-005-main+air+bone-lpips'
        DECODER_LORA = 'decoder+lora_{}.pkl'.format( sub_version )
        DECODER_AIR_LORA = 'cond_decoder+lora_air_{}.pkl'.format( sub_version )
        DECODER_BONE_LORA = 'cond_decoder+lora_bone_{}.pkl'.format( sub_version )
    elif version == 4:
        PRETRAINED_FOLDER = 'pretrained_weights/decoder/decoder-004-main-lpips'
        DECODER_LORA = 'decoder+lora_final.pkl'
        DECODER_AIR_LORA = None
        DECODER_BONE_LORA = None
    elif version == 3:
        PRETRAINED_FOLDER = 'pretrained_weights/decoder/decoder-003-main+air+bone-lpips'
        DECODER_LORA = 'decoder+lora_final.pkl'
        DECODER_AIR_LORA = 'cond_decoder+lora_air_final.pkl'
        DECODER_BONE_LORA = 'cond_decoder+lora_bone_final.pkl'
    elif version == 6:
        PRETRAINED_FOLDER = 'pretrained_weights/decoder/decoder-006-main-lpips+percept+cond'
        DECODER_LORA = 'decoder+lora_final.pkl'
        DECODER_AIR_LORA = None
        DECODER_BONE_LORA = None
    elif version == 7:
        PRETRAINED_FOLDER = 'pretrained_weights/decoder/decoder-007-main+lpips+percept+cond+discriminator'
        DECODER_LORA = 'decoder+lora_final.pkl'
        DECODER_AIR_LORA = None
        DECODER_BONE_LORA = None
    elif version == 8:
        PRETRAINED_FOLDER = 'pretrained_weights/decoder/decoder-009'
        DECODER_LORA = 'decoder+lora_final.pkl'
        DECODER_AIR_LORA = None
        DECODER_BONE_LORA = None
    elif version == 9:
        PRETRAINED_FOLDER = 'pretrained_weights/decoder/decoder-SynthRAD2023-001'
        DECODER_LORA = 'decoder+lora_synthrad2023.pkl'
        DECODER_AIR_LORA = None
        DECODER_BONE_LORA = None
    elif version == 10:
        PRETRAINED_FOLDER = 'pretrained_weights/decoder/decoder-SynthRAD2023-002'
        DECODER_LORA = 'decoder+lora-best-0510.pkl'
        DECODER_AIR_LORA = None
        DECODER_BONE_LORA = None
    elif version == 11:
        PRETRAINED_FOLDER = 'pretrained_weights/decoder/decoder-SynthRAD2023-003'
        DECODER_LORA = 'decoder+lora-best-0515.pkl'
        DECODER_AIR_LORA = None
        DECODER_BONE_LORA = None
        
    return

def decoder_init(
    flag_pretrained : str = 'default',
    version         : int = 5,
    final           : bool = True,
    dtype           : torch.dtype = torch.float32,
    ) -> Dict[ str, nn.Module ]:
    """
    decoder_init:
    初始化一組 conditional-refinement decoder 並載入相對應權重

    Args:
    ----------
    flag_pretrained: str
    指定要載入的 decoder 權重，目前支援以下模式：
    1. main:
        主幹，用於對來自 unet 的 latent-t0 解碼
    2. air:
        air 旁支，用於提供 air 的資訊
    3. bone:
        bone 旁支，用於提供 bone 的資訊
    4. default:
        僅回傳 stable-diffusion-v1.5 的預訓鍊權重，建議此時 lora_scale 設為 0
        以免隨機初始化的 lora module 干擾實際解碼資料

    Return:
    ----------
    Dict[ str, nn.Module ]
    一組 nn.Module ，包含 `decoder` 與量化用的 `post_quant_conv`

    """

    _version_set( version = version, final = final )

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
        "attention_processor" : "lora", 
    }
    decoder = Decoder( **arg_decoder )
    post_quant_conv = nn.Conv2d( 4, 4, 1 )
    
    for prefix in POSSIBLE_PATH_PREFIX:
        try:
            state_dict : Dict[ str, torch.Tensor ] = torch.load( '{}{}'.format( prefix, LORA_NAME ) )
            print( 'CRD_utils, decoder_init: pretrained VAE with name: `{}` loaded.'.format( LORA_NAME ) )
            break
        except:
            print( 'not found in {}{}'.format( prefix, LORA_NAME ) )
            continue
    
        raise FileNotFoundError("Can't find lora checkpoint.")
    
    decoder_param = {}
    post_quant_param = {}
    pattern_dec = r'decoder'

    for idx, name in enumerate( state_dict ):
        # 注意：個別宣告時，第一個前綴要拿掉
        # 因為那是在更上一層，以 AutoEncoderKL 為主體一起宣告時才有的名字
        name_split = name.split( '.' )
        new_name = '.'.join( name_split[ 1: ] )
        # decoder
        result = search( pattern_dec, name )
        if result:
            decoder_param[ new_name ] = state_dict[ name ]
            continue
        # post_quant_conv
        if 'post_quant_conv' in name_split:
            post_quant_param[ new_name ] = state_dict[ name ]

    # load state_dict
    if flag_pretrained is not None and flag_pretrained not in [ "main", "air", "bone", "default" ]:
        flag_pretrained = None
        print( 'CRD_utils, invalid `flag_pretrained` is given, set to None now.' )

    if flag_pretrained == 'default':
        ret = decoder.load_state_dict( decoder_param, False )
        print( 'CRD_utils: pretrained Stable Diffusion VAE loaded.' )
    elif flag_pretrained == 'main':
        ret = decoder.load_state_dict( torch.load( '{}/{}'.format( PRETRAINED_FOLDER, DECODER_LORA ) ) )
        print( 'CRD_utils, decoder_init: pretrained VAE with name: `{}` loaded.'.format( DECODER_LORA ) )
    elif flag_pretrained == 'air':
        ret = decoder.load_state_dict( torch.load( '{}/{}'.format( PRETRAINED_FOLDER, DECODER_AIR_LORA ) ) )
        print( 'CRD_utils, decoder_init: pretrained VAE with name: `{}` loaded.'.format( DECODER_AIR_LORA ) )
    elif flag_pretrained == 'bone':
        ret = decoder.load_state_dict( torch.load( '{}/{}'.format( PRETRAINED_FOLDER, DECODER_BONE_LORA ) ) )
        print( 'CRD_utils, decoder_init: pretrained VAE with name: `{}` loaded.'.format( DECODER_BONE_LORA ) )

    post_quant_conv.load_state_dict( post_quant_param )

    vae_dict = {
        "decoder" : decoder.to( dtype = dtype ).eval(),
        "post_quant_conv" : post_quant_conv.to( dtype = dtype ).eval()
    }

    return vae_dict

def _shape_check(
    input : torch.Tensor,
    target: int = 4,
    ):

    while( len( input.shape ) != target ):
        if len( input.shape ) > target:
            input = torch.squeeze( input, dim = 0 )
        elif len( input.shape ) < target:
            input = torch.unsqueeze( input, dim = 0 )
        else:
            raise ValueError( 'Invalid tensor shape with: {}'.format( input.shape ) )
    return input

def _device_check():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device 

def _dtype_check( emb : List[ torch.Tensor ], dtype = torch.float32 ):
    for t in emb:
        t = t.to( dtype = dtype )

    return emb

def _postprocess(
    dec : torch.Tensor,
    batch_size : int = 1
    ) -> Image.Image:
    image_processor = VaeImageProcessor( vae_scale_factor = 8 )
    norm_mask = [ True for cnt in range( batch_size ) ]

    ret : List[ Image.Image ] = image_processor.postprocess( dec, do_denormalize = norm_mask )
    return ret[ 0 ]

def decode_with_dict(
    latent          : torch.Tensor,
    dict_decoder    : Dict[ str, nn.Module ],
    dict_embed      : Dict[ str, List[ torch.Tensor ] ] = None,
    scales_cond     : List[ float ] = [ 0.05 ],
    lora_scale      : float = 0.1,
    scaling_factor  : float = 0.18215,
    dtype           : torch.dtype = torch.float32,
    device          : torch.device = 'cuda',
    flag_control    : bool = False,
    flag_postproc   : bool = True,
    ) -> Dict[ str, Any ]:
    """
    decode_with_dict:
    透過傳入的 decoder 進行解碼

    Args:
    ----------
    latent: torch.Tensor
    待解碼的 latent embedding
    ----------
    dict_decoder: Dict[ str, nn.Module ]
    來自 cond_refine_decoder_utils 宣告的 decoder dictionary，
    內含 `post_quant_conv` 與 `decoder`
    ----------
    dict_embed: Dict[ str, List[ torch.Tensor ] ]
    來自各個分支的 conditional guidance ，用於 main branch
    ----------
    scales_cond: List[ float ]
    用於控制每個 conditional guidance 對 main branch 的貢獻程度
    ----------
    lora_scale: float
    LoRA 模組的貢獻程度
    ----------
    flag_control: bool
    決定是否需要回傳 latent_buffer ，
    可以近似地認為是否為 condition branch 模式
    ----------
    flag_postproc: bool
    是否需要對輸出做後處理

    Return:
    ----------
    Dict: {
        `dec` : torch.Tensor, 解碼結果
        `img` : Image.Image | None, 後處理結果
        `cond_embedding` : List[ torch.Tensor ] | None, conditional guidance
    }
    """
    latent = _shape_check( input = latent, target = 4 )
    device = _device_check()
    if dict_embed is not None:
        for name in dict_embed:
            dict_embed[ name ] = _dtype_check( dict_embed[ name ] )

    post_quant_conv = dict_decoder[ 'post_quant_conv' ]
    decoder = dict_decoder[ 'decoder' ]
    ret_dict = {}

    post_quant_conv.to( device = device, dtype = dtype )
    decoder.to( device = device, dtype = dtype )

    latent = latent / scaling_factor
    z : torch.Tensor = post_quant_conv( latent.to( dtype = dtype, device = device ) )

    arg_decoder = {
        'z' : z,
        'latent_conds' : dict_embed, # 來自旁支 conditional-decoder 的各級輸入，預設為 None
        'scales_conds' : scales_cond, # 調節各旁支輸入的大小，預設為 None
        'lora_scale' : lora_scale, # 調節 mid-block 中 lora-attn 的 lora 權重，預設為 0.0
        'flag_control' : flag_control, 
        # 決定是否需要回傳 latent_buffer，
        # 也就是作為 conditional-decoder，預設為 False
    }
    
    if flag_control is True:
        dec : Tuple[ torch.Tensor, List[ torch.Tensor ] ] = decoder( **arg_decoder )
        dec, cond_embedding = dec
    elif flag_control is False:
        dec : torch.Tensor = decoder( **arg_decoder )
        cond_embedding = None

    if flag_postproc is True:
        dec = dec.cpu().detach().to( dtype = torch.float32 )
        img = _postprocess( dec )
    else:
        img = None
    
    if cond_embedding is not None:
        for i, emb in enumerate( cond_embedding ):
            cond_embedding[ i ] = emb.cpu().detach().to( torch.float16 )

    ret_dict = {
        'dec' : dec, # torch.Tensor, 解碼結果
        'img' : img, # Image.Image | None, 後處理影像
        'cond_embedding' : cond_embedding, # List[ torch.Tensor ] | None, 各級條件輸入
    }

    # GPU offload
    post_quant_conv.to( 'cpu' )
    decoder.to( 'cpu' )
    return ret_dict 

class CRD :
    def __init__( 
        self,
        mode_names : List[ str ] = [ 'main', 'air', 'bone' ],
        valid_names: List[ str ] = [ 'main', 'air', 'bone', 'default' ],
        dtype : torch.dtype = torch.float32,
        version : int = 6
        ) -> None:
        """
        Conditional Refinement Decoder
        DiffuCE 項目中的 Decoder ，用於將 latent 從 embedding 轉換成 pixel。
        這個過程中會參照來自原始影像的其他限制( Conditions )，
        詳細內容可參照 DiffuCE 項目。

        2024-01-06 更新：
        最新版 decoder 不再需要 fine-tune conditional guidance branch ，
        轉而 fine-tune 一組新的 main branch ，如此可以減輕所需的不同 branch parameter 數量，
        後續可透過輕量化 conditional branch 參數量大小而回歸舊設計

        2024-03-04 :
        若要載入原始的 decoder 就把 mode_names 填 `default`

        """
        self.decoders : Dict[ str, type[ Decoder ] ]= {}
        self.valid_names = valid_names
        self.mode_names = mode_names
        self.version = version

        for name in mode_names:
            if self.check_valid_names( name = name ):
                decoder_dict = decoder_init( flag_pretrained = name, version = version, dtype = dtype )
            else:
                decoder_dict = decoder_init( flag_pretrained = 'main', version = version, dtype = dtype )
            
            for module_name in decoder_dict:
                module = decoder_dict[ module_name ]
                decoder_dict[ module_name ] = module.eval()
            self.decoders[ name ] = decoder_dict

        self.cond_embs : Dict[ str, List[ torch.Tensor ] ] = {}

    def check_valid_names( 
        self, 
        name : str
        ) -> bool:

        if name in self.valid_names:
            return True
        else:
            return False
    
    def set_weight_by_state_dict(
        self,
        state_dict  : Dict[ str, torch.Tensor ],
        name        : str = 'main',    
        strict      : bool = True,
        ) -> bool:
        """
        set_weight_by_state_dict:
        透過外部傳入 state_dict 重設 weight ，預設為 eval()

        Args:
        ----------
        state_dict: Dict[ str, torch.Tensor ]
        decoder 的新參數
        ----------
        name: str
        decoder 的名稱，若不存在於 self.decoders 中會報錯，表示原本不存在這個 decoder
        ----------
        strict: bool
        用於 model.load_state_dict() 的參數，
        若為 True ，則嚴格要求所有參數大小一致

        Return:
        ----------
        Bool
        若成功就回傳 True ，否則 False

        """
        if name not in self.decoders.keys():
            # 當前要重設的 decoder 必須存在於 self.decoders 中
            return False
        
        # 透過 state_dict 重設 decoder
        self.decoders : Dict[ str, Dict[ str, nn.Module ] ]
        self.decoders[ name ][ 'decoder' ].load_state_dict( state_dict = state_dict, strict = strict )

        self.decoders[ name ][ 'decoder' ].eval()
        return True

    def set_cond_emb(
        self,
        latent      : Dict[ str, Union[ torch.Tensor, List[ torch.Tensor ] ] ],
        dtype       : torch.dtype = torch.float32,
        cond_lora   : float = 0.0
        ) -> None:
        """
        set_cond_emb:
        重設 CRD 的 condition embeddings

        Args:
        ----------
        latent: Dict[ str, Union[ torch.Tensor, List[ torch.Tensor ] ] ]
        以字典型式傳入的條件向量
        ----------
        dtype: torch.dtype
        ----------
        cond_lora: float
        旁支的 lora scale
        在 version 6 中為 0

        Return:
        ----------
        None

        """

        for name in latent:
            # 只處理 conditions
            if name == 'main':
                continue

            if name not in self.decoders.keys():
                # 若傳入的 condition name 不存在於當前 CRD 各分支中，
                # 則認為是直接走 Stable Diffusion V1.5 Decoder 的預訓練權重
                # 即是 main branch 搭配 LoRA scale = 0
                cond_lora = 0.0
                decoder_name = 'main'
            else:
                decoder_name = name

            
            if isinstance( latent[ name ], torch.Tensor ):
                # 若傳入的是 torch.Tensor ，表示需要進行 decode
                arg_cond_dec = {
                        "flag_control" : True, # condition branch 模式
                        "dict_embed" : None, # 無需 conditional guidance
                        "latent" : latent[ name ], # condition latent
                        "dict_decoder" : self.decoders[ decoder_name ], # decoder dict
                        "dtype" : dtype,
                        "lora_scale" : cond_lora,
                }
                self.cond_embs[ name ] = decode_with_dict( **arg_cond_dec )[ "cond_embedding" ]
            elif isinstance( latent[ name ], list ):
                # 若傳入的是 List[ torch.Tensor ]，表示已經在外部進行過 decode
                # 直接當作是 embedding 即可，反正不合就會在 self.decode() 使用時報錯
                self.cond_embs[ name ] = latent[ name ]
            else:
                # 以上皆非，報錯
                print( 'type of input: {}'.format( type( latent[ name ] ) ) )
                raise ValueError( 'Invalid latent type' )
        
        return 

    def decode( 
        self,
        latent          : Union[ 
                            Dict[ 
                                str, 
                                Union[ torch.Tensor, List[ torch.Tensor ] ] ], 
                            torch.Tensor ],
        scales_cond     : List[ float ] = [ 0.05 ],
        lora_scale      : float = 0.1,
        flag_control    : bool = False,
        output_type     : str = 'pil', # or `list_emb`, `dec`, `dict`
        dtype           : torch.dtype = torch.float32,
        output_dtype    : torch.dtype = torch.float16,
        mode            : str = 'train', # or `inference`
        cond_reset      : bool = True, # 是否重設條件
        ) -> Union[ Image.Image, List[ torch.Tensor ], torch.Tensor, Dict[ str, Any ] ]:
        """
        decode:
        由 CRD 單元進行解碼

        Args:
        ----------
        latent: Union[ torch.Tensor, Dict[ str, Union[ torch.Tensor, List[ torch.Tensor ] ] ]
        待解碼的 latent ，可以是單一 torch.Tensor 或是以 name 區隔的 dict
        ----------
        scales_cond: List[ float ]
        來自各旁支的權重大小
        ----------
        lora_scale: float
        當前分支上的 lora 大小
        ----------
        output_type: str
        要輸出的樣式，支援以下幾種：
        - `pil`: 後處理完畢的影像
        - `list_emb`: 用於其他旁支的 condition embedding
        - `dec`: 尚未後處理的 decoder 輸出
        - `dict`: 以 dictionary 將前三者包在一起輸出
        ----------
        dtype: torch.dtype
        CRD 運行的資料型態
        ----------
        output_dtype: torch.dtype
        輸出的資料型態

        Return:
        ----------
        Union[ torch.Tensor, Image.Image, List[ torch.Tensor ], Dict[ str, Any ] ]
        """

        
        if isinstance( latent, dict ):
            # 輸入為 dictionary
            # 應該含有 main 與 conditions
            # 這裡可以透過 `cond_reset` 決定各個條件是否需要重設
            for name in latent.keys():
                if ( name != 'main' and # 非主幹輸入
                    ( name not in self.cond_embs.keys() or cond_reset is True ) ): # 需重設條件或條件不存在
                    self.set_cond_emb( latent = { name : latent[ name ] } )
            
            if 'main' not in latent.keys():
                raise ValueError( 'Name `main` should exist in `latent`' )

            arg_main_dec = {
                "flag_control" : False, # 不需要 latent_buffer
                "dict_embed" : self.cond_embs, # 來自 condition branches 的 embedding
                "latent" : latent[ 'main' ], # main branch input
                "dict_decoder" : self.decoders[ 'main' ], # decoder parameters
                "lora_scale" : lora_scale, # lora module 強度
                "scales_cond" : scales_cond, # 來自各旁支的參數權重大小
                "dtype" : dtype,
            }
            result = decode_with_dict( **arg_main_dec )
        
        elif isinstance( latent, torch.Tensor ):
            # 輸入為 torch.Tensor 
            # 僅含有一個 latent ，預設為 main
            if len( self.cond_embs ) > 0 and flag_control is False:
                # 條件( self.cond_embs )存在且非控制旁支( flag_control is False )
                flag_control = False
                cond_embs = self.cond_embs
                
            else:
                # 否，這是個控制旁支操作
                flag_control = True
                cond_embs = None

                if self.version >= 6:
                    lora_scale = 0

            arg_dec = {
                "flag_control" : flag_control,
                "dict_embed" : cond_embs,
                "latent" : latent,
                "dict_decoder" : self.decoders[ 'main' ],
                "lora_scale" : lora_scale,
                "scales_cond" : scales_cond,
                "dtype" : dtype,
            }
            result : Dict[ str, Any ] = decode_with_dict( **arg_dec )
            if mode == 'inference':
                list_embedding : List[ torch.Tensor ] = result[ 'cond_embedding' ]
                for c, emb in enumerate( list_embedding ):
                    list_embedding[ c ] = emb.cpu().detach()
                result[ 'cond_embedding' ] = list_embedding
        else:
            raise ValueError("Invalid data type of latent")
        
        # output 
        if output_type == 'pil':
            img : Image.Image = result[ "img" ]
            return img
        elif output_type == 'list_emb':
            cond_embedding : List[ torch.Tensor ] = result[ "cond_embedding" ]
            for c, emb in enumerate( cond_embedding ):
                cond_embedding[ c ] = emb.to( dtype = output_dtype )
            return cond_embedding 
        elif output_type == 'dec':
            dec : torch.Tensor = result[ "dec" ]
            return dec
        elif output_type == 'dict':
            return result
        else:
            raise ValueError("Invalid return mode: {}".format( output_dtype ) )

        


