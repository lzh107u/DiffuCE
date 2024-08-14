import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
from PIL import Image
from typing import ( 
    Dict, List, Tuple,
    Any, Union, Sequence, Type,
    Optional, Callable )
from math import floor, ceil
from tqdm import tqdm
import pickle
import lpips
import random
from diffusers.image_processor import VaeImageProcessor

from vae_lora_test import (
    Decoder,
    Encoder,
    DBE,
    vae_dict_init,
)
from cond_refine_decoder_utils import CRD
from cddm import CDDM

from training_utils import show_history, figure_combine
from synthrad_utils import synthrad_pipeline, all_nii, pair_pipeline, check_size

from re import search

POSSIBLE_PATH_PREFIX = [
    'vae_lora_test/',
    '/home/zihong/CBCT/vae_lora_test/',
]

LATENT_ROOT = '0915_tuning/pickles'
OUTPUT_ROOT = '0915_tuning/CT'
LORA_NAME = 'lora_unit_epoch1000_mixed-ts.pt'

PRETRAINED_FOLDER = 'pretrained_weights/decoder/decoder-005-main+air+bone-lpips'
DECODER_LORA = 'decoder+lora_final.pkl'
DECODER_AIR_LORA = 'cond_decoder+lora_air_final.pkl'
DECODER_BONE_LORA = 'cond_decoder+lora_bone_final.pkl'

def parsing_latent( path : str ) -> Dict[ str, Any ]:
    """
    parsing_latent:
    回傳來自 0915_tuning/latent 中的 .pkl 檔
    每個 .pkl 檔都是一組輸入資料，包含與輸入 latent 相對應的 conditions 和用於計算 loss 的原始影像
    具體各項內容為：
        
        # ct: Image.Image
        # latent: torch.Tensor
        # cbct: Image.Image
        # conds: list[ Image.Image ]
        # mode_strength: Dict[ str, float ]
        # dtype: torch.dtype
    
    檔案由 latent_generation 生成而來

    Args:
    ---------
    path: str
    輸入一個 .pkl 的 path

    Return:
    ---------
    Dict[ str, Any ]
    當筆資料的所有內容
    """
    with open( path, 'rb' ) as f:
        result_dict : Dict[ str, Dict[ str, Any ] ] = pickle.load( f )
        # 唯一 key 是 'origin'

        # origin 內：
        # ct: Image.Image
        # latent: torch.Tensor
        # cbct: Image.Image
        # conds: list[ Image.Image ]
        # mode_strength: Dict[ str, float ]
        # dtype: torch.dtype
        
    return result_dict

class tuning_set( Dataset ):
    def __init__( 
        self,
        folder_root : str = LATENT_ROOT, 
        ) -> None:
        super().__init__()
        self.folder = glob( '{}/{}'.format( folder_root, '*.pkl' ) )
        self.folder.sort()
        self.image_processor = VaeImageProcessor( vae_scale_factor = 8 )

    def __getitem__( self, index : int ) -> Any:
        result_dict: Dict[ str, Any ] = parsing_latent( path = self.folder[ index % len( self.folder ) ] )
        seed = random.randint( a = 0, b = 39 )
        result_dict : Dict[ str, Union[ Image.Image, torch.Tensor, List[ np.ndarray ] ] ]
        
        result_dict[ 'ct' ] = self.image_processor.preprocess( result_dict[ 'ct' ].convert( "RGB" ) )
        result_dict[ 'cbct' ] = self.image_processor.preprocess( result_dict[ 'cbct' ].convert( "RGB" ) )
        result_dict[ 'idx' ] = torch.Tensor( [ index ] )
        aug : np.ndarray = result_dict[ 'aug_latents' ][ seed % len( result_dict[ 'aug_latents' ] ) ]
        aug : torch.Tensor = torch.from_numpy( aug )
        result_dict[ 'aug_latents' ] = aug

        for idx in range( len( result_dict[ 'conds' ] ) ):
            result_dict[ 'conds' ][ idx ] = self.image_processor.preprocess( result_dict[ 'conds' ][ idx ] )

        del result_dict[ 'mode_strength' ]
        return result_dict
    
    def __len__( self ) -> int:
        return len( self.folder )

class synthrad_set( Dataset ):
    def __init__(
        self,
        ) -> None:
        super().__init__()
        self.folder = all_nii()
        self.folder.sort()
        self.image_processor = VaeImageProcessor( vae_scale_factor = 8 )
    def __getitem__(
        self, 
        index
        ) -> Any:
        ct_size = check_size( index = index, content_idx = 1 )
        pos_slice = random.randint( 1, ct_size )
        fix_rotate = random.randint( -20, 20 )

        pair_dict = pair_pipeline( index = index, pos_slice = pos_slice, rand_rotate = False, fix_rotate = fix_rotate )
        ct : Dict[ str, Any ] = pair_dict[ "ct" ]
        cbct : Dict[ str, Any ] = pair_dict[ "cbct" ]

        while pair_dict is None or pair_dict[ "ct" ] is None or pair_dict[ "cbct" ] is None:
            pos_slice += 1
            pair_dict = pair_pipeline( index = index, pos_slice = pos_slice, rand_rotate = False, fix_rotate = fix_rotate )
            ct : Dict[ str, Any ] = pair_dict[ "ct" ]
            cbct : Dict[ str, Any ] = pair_dict[ "cbct" ]
        
        result_dict : Dict[ str, Union[ Image.Image, torch.Tensor, List[ np.ndarray ] ] ] = {}
        result_dict[ 'ct' ] = torch.squeeze( self.image_processor.preprocess( ct["images"][ 0 ].convert( "RGB" ) ), dim = 0 )
        result_dict[ 'cbct' ] = torch.squeeze( self.image_processor.preprocess( cbct["images"][ 0 ].convert( "RGB" ) ), dim = 0 )
        result_dict[ 'idx' ] = torch.Tensor( [ index ] )
        result_dict[ 'slice' ] = torch.Tensor( [ pos_slice ] )
        result_dict[ 'conds' ] = [ torch.squeeze( self.image_processor.preprocess( img.convert( "RGB" ) ), dim = 0 ) for img in ct["images"][ 1 : 4 ] ]

        return result_dict
    
    def __len__( self ) -> int:
        return len( self.folder )

def decoder_init(
    flag_pretrained : str = None
    ) -> Dict[ str, nn.Module ]:
    """
    
    """
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
            print( 'tune_decoder, decoder_init: pretrained VAE with name: `{}` loaded.'.format( LORA_NAME ) )
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
    if flag_pretrained is not None and flag_pretrained not in [ "main", "air", "bone" ]:
        flag_pretrained = None
        print( 'tune-decoder, invalid `flag_pretrained` is given, set to None now.' )

    if flag_pretrained is None:
        ret = decoder.load_state_dict( decoder_param, False )
    elif flag_pretrained == 'main':
        ret = decoder.load_state_dict( torch.load( '{}/{}'.format( PRETRAINED_FOLDER, DECODER_LORA ) ) )
        print( 'tune_decoder, decoder_init: pretrained VAE with name: `{}` loaded.'.format( DECODER_LORA ) )
    elif flag_pretrained == 'air':
        ret = decoder.load_state_dict( torch.load( '{}/{}'.format( PRETRAINED_FOLDER, DECODER_AIR_LORA ) ) )
        print( 'tune_decoder, decoder_init: pretrained VAE with name: `{}` loaded.'.format( DECODER_AIR_LORA ) )
    elif flag_pretrained == 'bone':
        ret = decoder.load_state_dict( torch.load( '{}/{}'.format( PRETRAINED_FOLDER, DECODER_BONE_LORA ) ) )
        print( 'tune_decoder, decoder_init: pretrained VAE with name: `{}` loaded.'.format( DECODER_BONE_LORA ) )

    post_quant_conv.load_state_dict( post_quant_param )

    vae_dict = {
        "decoder" : decoder,
        "post_quant_conv" : post_quant_conv
    }

    return vae_dict

def cond_loss( 
    pred        : torch.Tensor, 
    target      : torch.Tensor,
    idx         : int,
    folder_root : str = LATENT_ROOT,
    cond_idx    : int = 0,
    invert      : bool = False,
    dtype       : Optional[ torch.dtype ] = torch.float32,
    ) -> torch.Tensor:
    """
    cond_loss:
    condition pixelwise loss

    Args:
    ----------
    pred: torch.Tensor
    模型預測
    ----------
    target: torch.Tensor
    預測目標
    ---------
    idx: int
    目前資料在資料集內的位置
    ----------
    dtype: torch.dtype
    資料型態
    ---------
    folder_root: str
    子資料夾位置
    ---------
    cond_idx: int
    決定 condition mode 

    Return:
    ----------
    torch.Tensor 

    """
    # 透過 idx 從資料集中取得對應的資料
    folder = glob( '{}/{}'.format( folder_root, '*.pkl' ) )
    folder.sort()
    
    result_dict: Dict[ str, Any ] = parsing_latent( path = folder[ int( idx ) % len( folder ) ] )
    # 透過 condition index 取得對應的 condition mode
    cond : Image.Image = result_dict[ 'conds' ][ cond_idx ]
    # 轉換為 torch.Tensor
    # 注意：這裡會直接將通道從末端調至前端
    # => shape: n_ch x H x W
    transform = transforms.ToTensor()
    cond_mask = transform( cond )
    # 對齊 dimension, dtype & device
    cond_mask = cond_mask.unsqueeze( dim = 0 )
    cond_mask = cond_mask.to( device = device, dtype = dtype )

    # 對值域為 0-1 的 condition 進行黑白顛倒
    # 主要用於計算 air
    if invert is True:
        cond_mask = cond_mask - 1
        cond_mask = torch.abs( cond_mask )
    # 由 cond_mask ( 值域 0 ~ 1 )篩選要計算 loss 的位置
    # pred & target 都只留下黑色背景與骨骼位置的前景
    pred = pred * cond_mask
    target = target * cond_mask
    # 計算 pixelwise L2 loss
    loss : torch.Tensor = F.mse_loss( pred, target )
    
    return loss


"""
                ct_latent = fixed_encoder.encode( image = ct, mode = 'cond', dtype = dtype )
                ct_dec_dict : Dict[ str, Any ] = fixed_decoder.decode( latent = ct_latent, output_type = 'dict', output_dtype = dtype, dtype = dtype )
                
                air_latent = fixed_encoder.encode( image = conds[ 0 ], mode = 'cond', dtype = dtype )
                air_dec_dict : Dict[ str, Any ] = fixed_decoder.decode( latent = air_latent, output_type = 'dict', output_dtype = dtype, dtype = dtype )
                
                bone_latent = fixed_encoder.encode( image = conds[ 1 ], mode = 'cond', dtype = dtype )
                bone_dec_dict : Dict[ str, Any ] = fixed_decoder.decode( latent = bone_latent, output_type = 'dict', output_dtype = dtype, dtype = dtype )
                
                wavelet_latent = fixed_encoder.encode( image = conds[ 2 ], mode = 'cond', dtype = dtype ) # change to `wavelet`
                wavelet_dec_dict : Dict[ str, Any ] = fixed_decoder.decode( latent = wavelet_latent, output_type = 'dict', output_dtype = dtype, dtype = dtype )
                
                ct_embedding : List[ torch.Tensor ] = ct_dec_dict[ "cond_embedding" ]
                
                for i, emb in enumerate( ct_embedding ):
                    ct_embedding[ i ] = emb.to( dtype = torch.float32 ).detach()

                # 提取出用於 conditional decoder branches 的 embedding list
                air_embed : List[ torch.Tensor ] = air_dec_dict[ "cond_embedding" ]
                bone_embed : List[ torch.Tensor ] = bone_dec_dict[ "cond_embedding" ]
                wavelet_embed : List[ torch.Tensor ] = wavelet_dec_dict[ "cond_embedding" ]

                # 從 torch.float16 轉換至 torch.float32
                for i, emb in enumerate( air_embed ):
                    air_embed[ i ] = emb.to( dtype = torch.float32 ).detach()
                for i, emb in enumerate( bone_embed ):
                    bone_embed[ i ] = emb.to( dtype = torch.float32 ).detach()
                for i, emb in enumerate( wavelet_embed ):
                    wavelet_embed[ i ] = emb.to( dtype = torch.float32 ).detach()
"""

def pipeline_tune_decoder(
    dtype               : torch.dtype = torch.float16,
    scaling_factor      : float = 0.18215,
    n_epoch             : int = 1,
    accumulation_step   : int = 4,
    device              : torch.device = 'cuda',
    lora_scale          : float = 1.0,
    inference_cnt       : int = 3,
    inference_interval  : int = 10,
    date_postfix        : str = '1111',
    base_model : Optional[ str ] = 'stabilityai/stable-diffusion-2-base',
    ):
    """
    pipeline_tune_decoder
    """
    # 宣告各模組
    # encoder: for encoding conditions
    # decoder: training
    fixed_encoder = DBE( flag_enc_lora = False, ckpt_mode = 'default' ) # ckpt_mode 還可以選 'synthrad'
    # synth_encoder = DBE( flag_enc_lora = True, ckpt_mode = 'default' )
    fixed_decoder = CRD( dtype = dtype, mode_names = [ 'main' ], version = 8 )
    fixed_denoiser = CDDM( 
        dataset_mode = 'default', 
        pretrained_model_name_or_path = base_model, 
        lora_ckpt_dir = 'CDD_lora' )

    inference_decoder = CRD( dtype = dtype, mode_names = [ 'main' ], version = 8 )
    main_tuning_dict = decoder_init()
    discriminator = vae_dict_init( flag_enc_lora = True )

    # prepare dataset & dataloader
    training_set = tuning_set()
    # training_set = synthrad_set()
    training_loader = DataLoader( dataset = training_set, batch_size = 1, shuffle = True )
    # progress bar
    progress_bar = tqdm( total = len( training_loader ) )
    epoch_bar = tqdm( total = n_epoch )
    # declare parameters
    decoder : Decoder = main_tuning_dict[ 'decoder' ]
    discriminator : Encoder = discriminator[ 'encoder' ]
    
    post_quant_conv : nn.Conv2d = main_tuning_dict[ 'post_quant_conv' ]
    # prepare parameters
    decoder.to( dtype = torch.float32 )
    decoder.eval()
    decoder.mid_block.attentions[ 0 ].processor.train()

    discriminator.to( device = 'cpu', dtype = torch.float32 )
    discriminator.eval()
    discriminator.mid_block.attentions[ 0 ].processor.train()

    # 設定 trainable parameters & optimizer 
    param_lora = [ param for param in decoder.mid_block.attentions[ 0 ].processor.parameters() ]
    param_dis = [ param for param in discriminator.mid_block.attentions[ 0 ].processor.parameters() ]
    
    post_quant_conv.to( device = device, dtype = torch.float32 )
    post_quant_conv.eval()

    # reconstruction loss
    loss_history = []
    sub_histories = [ [], [], [], [], [] ]
    # lpips, air, bone, perceptual, discriminator

    optimizer = optim.Adam( param_lora, lr = 1e-5 )
    optim_dis = optim.Adam( param_dis, lr = 5*1e-6 ) # discriminator 的 learning rate 是 generator 的 1/2
    criterion_dis = nn.BCELoss()
    real_answer = 1.0
    fake_answer = 0.0
    func_sigmoid = nn.Sigmoid()
    func_pooling = nn.AvgPool3d( kernel_size = ( 8, 64, 64 ), stride = 1, padding = 0 )

    image_processor = VaeImageProcessor( vae_scale_factor = 8 )
    # 這裡選擇 alex 作為 LPIPS 指標計算的特徵提取器
    # 參考：https://github.com/richzhang/PerceptualSimilarity
    loss_fn_lpips : nn.Module = lpips.LPIPS( net = 'alex' )
    loss_fn_lpips.to( device = device )
    best_loss = None

    epoch_bar.clear()
    epoch_bar.reset()
    for epoch in range( n_epoch ):
        # 重置進度條
        progress_bar.clear()
        progress_bar.reset()
        # 重置累計參數
        running_loss = 0.0
        running_record = [ 0.0, 0.0, 0.0, 0.0, 0.0 ]
        # 清空優化器
        optimizer.zero_grad()
        for epoch_idx, batch in enumerate( training_loader ):
            if epoch_idx > 3:
                break
            batch : Dict[ str, Any ]
            # extract data
            ct : torch.Tensor = batch[ 'ct' ]       # 經過前處理的 ct 影像
            cbct : torch.Tensor = batch[ 'cbct' ] # 經過前處理的 cbct 影像
            conds : List[ torch.Tensor ] = batch[ 'conds' ] # 經過前處理的 conditions 
            
            # ct = torch.squeeze( ct, dim = 0 )

            # --------------------------------------------------------
            # 注意：這裡的 aug_latent 指的是來自 UNet(CDD) 的 embedding
            # 且確實可以注意到這裡真的過了一次 denoiser
            # --------------------------------------------------------
            # aug_latent : torch.Tensor = synth_encoder.encode( image = cbct, timestep = 200, mode = 'main' )
            aug_latent : torch.Tensor = fixed_encoder.encode( image = cbct, timestep = 200, mode = 'main' )
            aug_latent : torch.Tensor = fixed_denoiser.denoise(
                image = aug_latent,
                cond = conds,
                latent = aug_latent,
                scales = [ 0.05, 0.05, 0.3 ],
            )
            aug_latent = aug_latent.detach()
            
            indice : torch.Tensor = batch[ 'idx' ]
            
            # ---------------------------------------------------------------------------------------------
            # 運算順序：
            # 1. 透過 moment 重參數化一組 gt
            # 2. 透過 cond_moment 重參數化一組 condition embedding
            # 3. 由 condition embedding 與 decoder 計算出用於 decoder main branch 各級 module 的 condition guidance
            # 4. 由來自 unet 的 latent 與 condition guidance 解碼出 main branch 的結果用於計算 loss 與 metric
            # ---------------------------------------------------------------------------------------------
            
            # -------------------------------------------------------------------------------
            # Phase 1. : encode sequence
            # 透過 encoder 得出：air, bone 與 ct 的 latent
            # 其中 air, bone 的 latent 將送進 SD v1.5 decoder 得出 conditional embedding 
            # 用於 main branch 的預測
            # 另外 ct latent 則用於作為 gt 與 main branch 計算 loss
            # 注意：在 Phase 1. 中，為了降低計算成本，dtype 設為 torch.float16
            # -------------------------------------------------------------------------------
            with torch.no_grad():
                # 第一階段統一採用 torch.float16 以降低計算成本
                dtype = torch.float16
                # 透過 ct 算出 ground truth
                # 透過 conds 算出 conditional guidance embeddings

                def prepare_emb( image: torch.Tensor ) -> Tuple[ Dict[ str, Any ], List[ torch.Tensor ] ]:
                    # 先由 encoder 給出在 latent space 中的 embedding
                    latent = fixed_encoder.encode( image = image, mode = 'cond', dtype = dtype )
                    # 再由 decoder 給出每一個 block 可以用的 intermediate embeddings
                    dec_dict : Dict[ str, Any ] = fixed_decoder.decode( 
                        latent = latent, 
                        output_type = 'dict', 
                        output_dtype = dtype, 
                        dtype = dtype,
                        mode = 'inference' )
                    embedding : List[ torch.Tensor ] = dec_dict[ "cond_embedding" ]
                    # 將每一層 intermediate embedding 都轉換成 fp32 並 detach
                    # 畢竟分支 network 都沒有要 train ，只是作為 embedding provider，真正的訓練對象只有 main branch
                    for i, emb in enumerate( embedding ):
                        embedding[ i ] = emb.to( dtype = torch.float32 ).detach()
                    
                    return dec_dict, embedding
                # -------------------------------------------------------
                # ct: ground truth; air, bone, wavelet: condition branch
                # 這裡決定有甚麼 condition 參與 decode
                # 若要調整，就把新的條件以 PIL 影像傳入 VAEImageProcessor 後轉成 torch.Tensor
                # 再丟進 prepare_emb，
                # 並把第二個回傳參數放進 `dict_cond_embedding`
                # -------------------------------------------------------
                ct_dec_dict, ct_embedding = prepare_emb( image = ct )
                _, air_embed = prepare_emb( image = conds[ 0 ] )
                _, bone_embed = prepare_emb( image = conds[ 1 ] )
                _, wavelet_embed = prepare_emb( image = conds[ 2 ] )
                
                # ct_gt: 來自標準答案的 embedding ，用於評估 loss
                ct_gt : torch.Tensor = ct_dec_dict[ "dec" ]
                
                ct_gt = ct_gt.to( device = device, dtype = torch.float32 )
                ct_gt = ct_gt.detach() # 這裡透過 detach 移除後續 backward 影響
                
                # dict_cond_embedding: 用於 main branch 的 condition embedding
                """
                dict_cond_embedding = {
                    "air" : air_embed,
                    "bone" : bone_embed,
                    "cbct" : cbct_embed,
                }
                """
                dict_cond_embedding = { "wavelet" : wavelet_embed }
            # Phase 1. outputs: 
            # dict_cond_embedding: Dict[ str, List[ torch.Tensor ] ]，用於 decoder
            # ct_gt: torch.Tensor，用於評估 loss

            # ---------------------------------------------------------------------------------------------
            # Phase 2. : main decoding part
            # 由來自 unet 的 latent 與 condition guidance 解碼出 main branch 的結果用於計算 loss 與 metric
            # 這裡將 dict_cond_embedding 送進去協助 decode
            # 第二階段中為了訓練精度，統一使用 torch.float32
            # ---------------------------------------------------------------------------------------------
            dtype = torch.float32
            if len( aug_latent.shape ) > 4:
                aug_latent = torch.squeeze( aug_latent, dim = 0 )
            aug_latent = aug_latent / scaling_factor
            z : torch.Tensor = post_quant_conv( aug_latent.to( dtype = dtype, device = device ) )
            
            arg_decoder = {
                "z" : z,
                "latent_conds" : dict_cond_embedding,
                "scales_conds" : [ 1.0 ], # 訓練時各條件引導強度皆為 1.0
                "lora_scale" : lora_scale,
                "flag_control" : True, # 這一項是表示是否需要 latent_buffer
            }
            decoder.to( device = 'cuda' )
            ret : Tuple[ torch.Tensor, List[ torch.Tensor ] ] = decoder( **arg_decoder )
            dec, main_embedding = ret
            # Phase 2. outputs:
            # dec: torch.Tensor，解碼樣本，尚未透過後處理成為 PIL.Image
            # main_embedding: List[ torch.Tensor ]，main branch 各 block 間的 embedding ，用於評估 Perceptual Loss

            # --------------------
            # Phase 3. : 計算損失
            # --------------------

            # discriminator 損失：在真實樣本上的損失 + 在生成樣本上的損失
            # 皆與真正的答案進行評估，真樣本與 real_answer ；假樣本與 fake_answer，目的是準確判斷出真偽
            discriminator.to( device = 'cuda' )
            # stage 1: 真實樣本損失
            arg_discriminator = {
                'x' : ct_gt,
                'lora_scale' : 1.0,
            }
            pred_real : torch.Tensor = discriminator( **arg_discriminator ) # 判別真實影像
            dis_pred_real : torch.Tensor = func_sigmoid( pred_real ) # 給出 0-1 間的答案
            dis_pred_real : torch.Tensor = func_pooling( dis_pred_real )
            real_ans = torch.full( ( 1, ), fill_value = real_answer, dtype = torch.float32, device = 'cuda' ) # 真實答案
            loss_dis_real : torch.Tensor = criterion_dis( dis_pred_real.squeeze().unsqueeze( dim = 0 ), real_ans ) # 計算判別器在真實樣本上的損失
            loss_dis_real.backward()
            # stage 2: 生成樣本損失
            arg_discriminator = {
                'x' : dec.detach(), # 注意這裡 detach ，與生成器無關。目的是塑造出代表 real 與 fake 的 scaler ，與任何其他網路都無關
                'lora_scale' : 1.0,
            }
            pred_fake : torch.Tensor = discriminator( **arg_discriminator ) # 判別生成影像
            dis_pred_fake : torch.Tensor = func_sigmoid( pred_fake ) # 給出 0-1 間的答案
            dis_pred_fake : torch.Tensor = func_pooling( dis_pred_fake )
            fake_ans = torch.full( ( 1, ), fill_value = fake_answer, dtype = torch.float32, device = 'cuda' ) # 生成答案
            loss_dis_fake : torch.Tensor = criterion_dis( dis_pred_fake.squeeze().unsqueeze( dim = 0 ), fake_ans ) # 計算判別器在生成樣本上的損失
            loss_dis_fake.backward()

            # generator ( CRD ) 損失：生成樣本當成真實影像的損失
            # 注意這裡與真實影像評估的是 real_answer 而非 fake_answer
            # 因為 CRD 是以 `真實` 為目的學習的
            arg_discriminator = {
                'x' : dec, # 注意這裡就不 detach，讓梯度流回 CRD
                'lora_scale' : 1.0, 
            }
            pred_gen : torch.Tensor = discriminator( **arg_discriminator ) # 判別生成影像
            dis_pred_gen : torch.Tensor = func_sigmoid( pred_gen ) # 給出 0-1 間的答案
            dis_pred_gen : torch.Tensor = func_pooling( dis_pred_gen )
            loss_dis_gen : torch.Tensor = criterion_dis( dis_pred_gen.squeeze().unsqueeze( dim = 0 ), real_ans ) # 計算生成器 & 判別器在生成樣本上的損失
            
            loss_dis = loss_dis_real + loss_dis_fake

            # 計算 lpips
            dec_norm : torch.Tensor = image_processor.postprocess( image = dec, output_type = 'pt', do_denormalize = [ True ] )
            dec_gt : torch.Tensor = image_processor.postprocess( image = ct_gt, output_type = 'pt', do_denormalize = [ True ] )
            loss_lpips : torch.Tensor = loss_fn_lpips( dec_norm, dec_gt )
            # 計算 air loss
            arg_cond_loss = {
                "pred" : dec,
                "target" : ct_gt,
                "idx" : indice[ 0 ][ 0 ].item(),
                "cond_idx" : 0, # 透過 cond_mode 將 condition 指定為 bone
                "invert" : True, # air 需要黑白反轉
            }
            loss_air = cond_loss( **arg_cond_loss )
            # 計算 bone loss
            arg_cond_loss[ "cond_idx" ] = 1
            arg_cond_loss[ "invert" ] = False # bone 不需要黑白反轉
            loss_bone = cond_loss( **arg_cond_loss )
            # 計算 perceptual loss
            loss_percept = F.mse_loss( main_embedding[ 0 ], ct_embedding[ 0 ] )
            # 總和： LPIPS + air + bone + perceptual loss
            air_scale = loss_percept.item() / loss_air.item()
            bone_scale = loss_percept.item() / loss_bone.item()
            lpips_scale = loss_percept.item() / loss_lpips.item()
            total_loss = lpips_scale * loss_lpips + air_scale*loss_air + bone_scale*loss_bone + loss_percept + loss_dis_gen
            # 計算梯度
            total_loss.backward()
            # 若目前 epoch 是 accumulation_step 的整數倍
            # 就進行一次向後傳播
            if ( epoch_idx + 1 ) % accumulation_step == 0:
                optimizer.step() # back propagation
                optimizer.zero_grad() # clean optimizer

                optim_dis.step() # discriminator update
                optim_dis.zero_grad() # clean optimizer
            
            running_loss += total_loss.item()
            for c, l in enumerate( [ loss_lpips, loss_air, loss_bone, loss_percept, loss_dis ] ):
                running_record[ c ] += l.item()
            progress_bar.update( 1 )

            
        # end of single epoch
        running_loss = running_loss / len( training_loader )
        if best_loss is None:
            best_loss = running_loss
        elif best_loss > running_loss:
            best_loss = running_loss
            torch.save( decoder.state_dict(), 'decoder+lora-best-{}.pkl'.format( date_postfix ) )
            
        loss_history.append( running_loss )
        for c, l in enumerate( running_record ):
            l = l / len( training_loader )
            sub_histories[ c ].append( l )

        epoch_bar.update( 1 )

        # -------------------------------------
        # Inference M times for each N epoches
        # -------------------------------------
        if ( epoch + 1 ) % inference_interval != 0:
            continue
        for c in range( inference_cnt ):
            with torch.no_grad():
                dtype = torch.float16
                state_dict = decoder.state_dict()
                inference_decoder.set_weight_by_state_dict( state_dict = state_dict, name = 'main', strict = True )
                inference_index = random.randint( a = 0, b = len( training_set ) )
                data_dict : Dict[ str, Any ] = training_set[ inference_index % len( training_set ) ]

                
                conds : List[ torch.Tensor ] = data_dict[ 'conds' ] # 經過前處理的 conditions 
                cbct : torch.Tensor = data_dict[ 'cbct' ].unsqueeze( dim = 0 )
                ct : torch.Tensor = data_dict[ 'ct' ].unsqueeze( dim = 0 )
                conditions : List[ torch.Tensor ] = [ cond.unsqueeze( dim = 0 ) for cond in conds ]

                air_latent = fixed_encoder.encode( image = conds[ 0 ].unsqueeze( dim = 0 ), mode = 'cond', dtype = dtype )
                bone_latent = fixed_encoder.encode( image = conds[ 1 ].unsqueeze( dim = 0 ), mode = 'cond', dtype = dtype )
                cbct_latent = fixed_encoder.encode( image = conds[ 2 ].unsqueeze( dim = 0 ), mode = 'cond', dtype = dtype )
                latent = fixed_encoder.encode( image = cbct, mode = 'main', timestep = 200 )
                # latent : torch.Tensor = data_dict[ 'aug_latents' ] # 經過 latent UNet denoise 的結果
                
                latent : torch.Tensor = fixed_denoiser.denoise(
                    image = latent,
                    cond = conditions,
                    latent = latent,
                    scales = [ 0.05, 0.05, 0.3 ],
                )

                latent_dict = {
                    'main' : latent,
                    'cbct' : cbct_latent,
                }

                inference_image : Image.Image = inference_decoder.decode( 
                    latent = latent_dict, 
                    scales_cond = [ 0.2 ], ## air, bone, wavelet
                    lora_scale = 1.0, 
                    flag_control = False, 
                    mode = 'inference',
                    cond_reset = True,
                    output_type = 'pil' )
                
                raw_output : Image.Image = inference_decoder.decode( latent = latent, lora_scale = 0.0, flag_control = True, output_type = 'pil' )
                inference_dict = {
                    'CT-{}'.format( inference_index ) : image_processor.postprocess( image = ct, output_type = 'pil', do_denormalize = [ True ] )[ 0 ],
                    'CBCT-{}'.format( inference_index ) : image_processor.postprocess( image = cbct, output_type = 'pil', do_denormalize = [ True ] )[ 0 ],
                    'output' : inference_image,
                    'w/o decoder' : raw_output
                }
                
                plotname = 'inference-{}-epoch{}-index{}'.format( date_postfix, epoch, inference_index )
                figure_combine( images = inference_dict, figname = '{}.png'.format( plotname ), plotname = plotname )
        arg_history = {
            "data" : {
                "loss-lpips" : sub_histories[ 0 ],
                "loss-air" : sub_histories[ 1 ],
                "loss-bone" : sub_histories[ 2 ],
                "loss-percept" : sub_histories[ 3 ],
                "loss-discriminator" : sub_histories[ 4 ],
            },
            "title" : "loss over {} epoch".format( n_epoch ),
            "ylabel" : "loss val",
            "filename" : "loss_history_decoder_{}_sub".format( date_postfix ),
        }
        show_history( **arg_history )


    # end of epoches
    progress_bar.close()
    epoch_bar.close()

    arg_history = {
        "data" : {
            "loss-lpips" : sub_histories[ 0 ],
            "loss-air" : sub_histories[ 1 ],
            "loss-bone" : sub_histories[ 2 ],
            "loss-percept" : sub_histories[ 3 ],
            "loss-discriminator" : sub_histories[ 4 ],
        },
        "title" : "loss over {} epoch".format( n_epoch ),
        "ylabel" : "loss val",
        "filename" : "loss_history_decoder_{}_sub".format( date_postfix ),
    }
    show_history( **arg_history )
    arg_history[ "data" ] = loss_history
    arg_history[ "filename" ] = "loss_history_decoder_{}".format( date_postfix )
    show_history( **arg_history )

    torch.save( decoder.state_dict(), 'decoder+lora_default_{}.pkl'.format( date_postfix ) )
    
    return

def pipeline_inference(
    dtype               : torch.dtype = torch.float32,
    scaling_factor      : float = 0.18215,
    device              : torch.device = 'cuda',
    lora_scale          : float = 1.0,
    valid_idx           : int = 5,
    scales_cond         : List[ float ] = [ 0.05 ],
    input_mode          : str = 'ct',
    version : int = 6,
    ):
    # 宣告各模組
    # encoder: for encoding conditions
    # decoder: training
    encoder = DBE()
    decoder = CRD( version = 6 )
    # prepare dataset & dataloader
    valid_set = tuning_set()

    image_processor = VaeImageProcessor( vae_scale_factor = 8 )

    if input_mode not in [ 'ct', 'cbct' ]:
        input_mode = 'ct'

    with torch.no_grad():
        if input_mode == 'ct':
            data_dict : Dict[ str, Any ] = valid_set[ valid_idx ]
            # ct: torch.Tensor
            # latent: torch.Tensor
            # cbct: torch.Tensor
            # conds: list
            # ct_moments: torch.Tensor
            # conds_moments: list
            # idx: torch.Tensor
            latent : torch.Tensor = data_dict[ 'latent' ]
            conds : List[ Image.Image ] = data_dict[ 'conds' ]
            ct : torch.Tensor = data_dict[ 'ct' ]
            cbct : torch.Tensor = data_dict[ 'cbct' ]

        
        
        # 給出條件
        # air & bone
        cond_order = [ 'air', 'bone' ]
        dict_embedding = {}
        for cond_idx in range( 2 ):

            cond_latent = encoder.encode( image = conds[ cond_idx ], mode = 'cond' )
            
            dict_embedding[ cond_order[ cond_idx ] ] = cond_latent

        # 將來自 unet 的 latent 搭配 condition embeddings 進行解碼
        
        dict_embedding[ "main" ] = latent
        ct_no_lora = decoder.decode( latent = latent, lora_scale = 0 )
        images = {}
        result : List[ Image.Image ] = image_processor.postprocess( ct, do_denormalize = [ True ] )
        images[ "origin" ] = result[ 0 ]
        result : List[ Image.Image ] = image_processor.postprocess( cbct, do_denormalize = [ True ] )
        images[ "cbct" ] = result[ 0 ]
        images[ "no_lora" ] = ct_no_lora
        
        def decode_sequence( 
            list_lora_scale: List[ float ], 
            list_cond_scale: List[ float ],
            results : Dict[ str, Any ] = {} 
            ) -> Dict[ str, Image.Image ]:
            print( 'decoder sequence starts ...' )
            for s_lora in list_lora_scale:
                for s_air in list_cond_scale:
                    for s_bone in list_cond_scale:
                        ret : Image.Image = decoder.decode( latent = dict_embedding, scales_cond = [ s_air, s_bone ], lora_scale = s_lora )

                        results[ "lora{}-air{}-bone{}".format( s_lora, s_air, s_bone ) ] = ret

            return results
        
        images = decode_sequence( [ 0.2 ], [ 0.01, 0.05, 0.1 ], images )
    

    figure_combine( images, figname = 'tuned-decoder-{}-version{}.png'.format( valid_idx, version ), plotname = 'tuned-decoder-lora_scale-cond_scale' )
    
    return 

def pipeline_dataset( epoch : int = 200 ):
    testset = synthrad_set()
    testloader = DataLoader( testset, batch_size = 1 )
    progressbar = tqdm( total = epoch )
    batchbar = tqdm( total = len( testloader ) )
    batchbar.clear()
    for epc in range( epoch ):
        
        for batch in testloader:
            tensors : torch.Tensor = batch[ 'cbct' ]
            index : torch.Tensor = batch[ 'idx' ]
            pos_slice : torch.Tensor = batch[ 'slice' ]
            # print( 'size of tensor: {}'.format( tensors.shape ) )
            if len( tensors.shape ) != 4:
                print( "Warning: invalid shape: {} with index: {}, slice: {}".format( tensors.shape, index, pos_slice ) )
            
            batchbar.update( 1 )

        batchbar.clear()
        batchbar.reset()

        progressbar.update( 1 )


def pipeline_preprocess( dtype = torch.float32 ):
    folder = glob( '{}/*.pkl'.format( LATENT_ROOT ) )
    progress = tqdm( total = len( folder ) )
    progress.clear()
    progress.reset()

    encoder = DBE( flag_enc_lora = False )
    decoder = CRD( dtype = torch.float32 )
    with torch.no_grad():
        for cnt, path in enumerate( folder ):
            result_dict : Dict[ str, Any ] = parsing_latent( path = path )

            ct : Image.Image = result_dict[ 'ct' ]
            conds : List[ Image.Image ] = result_dict[ 'conds' ]
            ct_latent : torch.Tensor = result_dict[ 'latent' ]

            ct_dec_emb : List[ torch.Tensor ] = decoder.decode( latent = ct_latent.to( dtype = dtype ), output_type = 'list_emb', output_dtype = torch.float16 )
            
            cond_latents : List[ torch.Tensor ] = []
            for cond in conds:
                cond_emb = encoder.encode( image = cond )
                cond_emb = cond_emb.cpu().detach().to( torch.float16 )
                cond_latents.append( cond_emb )
        
            with open( '{}/{}.pkl'.format( OUTPUT_ROOT, cnt ), 'wb' ) as f:
                if 'ct_dec_emb' in result_dict.keys():
                    del result_dict[ 'ct_dec_emb' ]
                if 'cond_dec_embs' in result_dict.keys():
                    del result_dict[ 'cond_dec_embs' ]
                output = {}
                result_dict[ 'cond_latents' ] = cond_latents
            
                output[ 'origin' ] = result_dict
                pickle.dump( output, f )

            progress.update( 1 )
            
    progress.close()

def inspect(
    idx = 30,
    folder_root : str = LATENT_ROOT,
    cond_idx    : int = 0, # 用 air
    dtype       : Optional[ torch.dtype ] = torch.float32,
    ):
    
    # 透過 idx 從資料集中取得對應的資料
    folder = glob( '{}/{}'.format( folder_root, '*.pkl' ) )
    folder.sort()
    
    result_dict: Dict[ str, Any ] = parsing_latent( path = folder[ int( idx ) % len( folder ) ] )
    # 透過 condition index 取得對應的 condition mode
    cond : Image.Image = result_dict[ 'conds' ][ cond_idx ]
    # 轉換為 torch.Tensor
    # 注意：這裡會直接將通道從末端調至前端
    # => shape: n_ch x H x W
    transform = transforms.ToTensor()
    cond_mask = transform( cond )
    # 對齊 dimension, dtype & device
    cond_mask = cond_mask.unsqueeze( dim = 0 )
    cond_mask = cond_mask.to( device = device, dtype = dtype )

    target = torch.zeros_like( cond_mask )
    
    
    return 
    # 由 cond_mask ( 值域 0 ~ 1 )篩選要計算 loss 的位置
    # pred & target 都只留下黑色背景與骨骼位置的前景
    pred = pred * cond_mask
    target = target * cond_mask
    # 計算 pixelwise L2 loss
    loss : torch.Tensor = F.mse_loss( pred, target )


    return 
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print( 'device: {}'.format( device ) )
    arg_train = {
        "n_epoch" : 2,
        "accumulation_step" : 2,
        "device" : device,
        "dtype" : torch.float16,
        "inference_interval" : 12,
        "date_postfix" : "0808",
        'base_model' : 'runwayml/stable-diffusion-v1-5'
    }
    arg_valid = {
        "valid_idx" : 32,
        "device" : device,
        "lora_scale" : 0.3,
        "scales_cond" : [ 0.05 ],
    }
    # pipeline_preprocess( dtype = torch.float32 )
    pipeline_tune_decoder( **arg_train )
    # pipeline_dataset()
    # pipeline_inference( **arg_valid )
    # inspect()