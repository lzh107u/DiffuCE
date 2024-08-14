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
    Dict, List, Tuple, Callable, Any, Union, Sequence, Type, Optional )
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from random import randint

import diffusers
from diffusers import (
    DDPMScheduler,
    DDIMScheduler
)
from vae_lora_test import AutoencoderKL
AutoencoderKL : Type[ diffusers.AutoencoderKL ]
from diffusers.image_processor import VaeImageProcessor
from dicom_utils import (
    single_wavelet_2d,
    to_uint8_255, 
)
from synthrad_utils import (
    synthrad_pipeline
)

from training_utils import write_file
"""
關於 AutoencoderKL:

"""

from dicom_utils import thresh_mask, to_uint8_255, normalize, find_body

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'
weight_dtype = torch.float32 # 註：使用 fp16 會導致 inf 錯誤
resolution = 512

# 從 PIL.Image 到 torch.Tensor 的 pipeline
image_augment_transforms = transforms.Compose(
        [
            transforms.Resize( resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomRotation( degrees = ( -30, 30 ) ),
            transforms.RandomHorizontalFlip(),
        ]
    )
image_transforms = transforms.Compose(
    [
        transforms.Resize( resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop( resolution ),
        transforms.ToTensor(),
        transforms.Normalize( [ 0.5 ], [ 0.2 ] ),
    ]
)

CT_DATASET_PATH = '0821_tuning/CT'
CBCT_DATASET_PATH = '0821_tuning/CBCT'
# PICKLE_DATASET_PATH = '0915_tuning/CBCT'
PICKLE_DATASET_PATH = '0915_tuning/pickles'
TUNING_SET_PREFIX = 'DBE_tuning'
SYNTHRAD2023_PATH = 'SynthRAD2023/pelvis'

class AlignSet( Dataset ):
    def __init__( 
        self, 
        boundary : float = 300, 
        mode : str = 'train',
        source : str = 'synthrad' ) -> None:
        """
        AlignSet:

        Init Args:
        ----------
        boundary: float, default: 300
        資料集涵蓋目標檔案數量，若低於 1 ，表示百分比，若高於 1 ，表示實際數量
        ----------
        mode: str, default: `train`
        若模式為 `train` ，會對影像進行前處理

        """
        super().__init__()

        # 確定檔案讀取方式
        if source not in [ 'synthrad', 'dicom', 'pickle' ]:
            source = 'synthrad'
        self.source = source

        # 設定檔案位置
        if source == 'synthrad':
            self.dataset_len = len( glob( SYNTHRAD2023_PATH + '/*' ) )
        elif source == 'pickle':
            self.dataset_names = glob( '{}/{}'.format( PICKLE_DATASET_PATH, '*.pkl' ) )
            self.dataset_len = len( self.dataset_names )
            
        elif source == 'dicom':
            raise ValueError("The `dicom` mode hasn't implemented.")
        

        if boundary < 0 or boundary >= self.dataset_len:
            print( 'AlignSet, Warning: Invalid index boundary.' )
            print( 'Expect in range: float[ 0, 1 ] or int[ 1, 400 ]' )
            print( 'Set to 0.5' )
            boundary = 0.5

        if boundary < 1 and boundary > 0:
            self.boundary = math.floor( self.dataset_len * boundary )
            
        elif boundary >= 1:
            self.boundary = math.floor( boundary )
        self.mode = mode

    def __getitem__(self, index: int ) -> Tuple[ torch.Tensor, torch.Tensor ]:
        # 確定當前模式
        if self.mode == 'train':
            index = index % self.boundary
        else:
            index = index % ( self.dataset_len - self.boundary ) + self.boundary

        if self.source == 'synthrad':
            pos_slice = randint( 5, 90 )
            cbct_images = synthrad_pipeline( index = index, content_idx = 0, pos_slice = pos_slice )
            ct_images = synthrad_pipeline( index = index, content_idx = 1, pos_slice = pos_slice )

            rotated_images = self._preprocessing(
                ct_pil = ct_images[ 0 ],
                cbct_pil = cbct_images[ 0 ],
                cond_air = cbct_images[ 1 ],
                cond_bone = cbct_images[ 2 ],
            )
            return rotated_images[ 0 ], rotated_images[ 1 ]

        # 讀取檔案( pickle & image )
        # pickle 模式是讀取之前整理出來的 CT 影像，將經過前處理的 CT, CBCT 與 Conditions
        # 直接儲存進 pickle 檔中，省去重複處理的時間
        # __getitem__ 的最終目的就是找出正確的 `ct_pil` 與 `cbct_pil`，不管是來自
        # pickle 檔還是 synthrad_pipeline 都一樣

        filename = self.dataset_names[ index % self.dataset_len ]
        with open( filename, 'rb' ) as file:
            ct_dict = pickle.load( file )
            # name: str => dicom name
            # hu: np.ndarray => hu value of ct
            # cbct: Image.Image => sparse ver. of ct

        ct_pil : Image.Image = ct_dict[ 'ct' ]
        cbct_pil : Image.Image = ct_dict[ 'cbct' ]

        # 0915 實驗設定
        # cbct_pil 將改為 cbct 的二次小波低頻信號
        # LL, HH = single_wavelet_2d( cbct_pil.convert( 'L' ) )
        # cbct_pil, HH = single_wavelet_2d( LL )
        # cbct_pil = Image.fromarray( to_uint8_255( cbct_pil ) ).resize( ( 512, 512 ) )

        # 基於 hu 值產生 conditions 
        # conditions = self._hu_condition( hu = ct_dict[ 'hu' ] )
        # 2024-08-06: 
        # 在新的 pickle 檔中已經不存在 hu value，因此取消 self._hu_condition()
        # 直接基於 pickle 內的 conditions
        conditions = ct_dict[ 'conds' ][ 0 : 2 ] # 分別是 air 與 bone
        
        rotated_images = self._preprocessing( 
            cond_air = conditions[ 0 ], 
            cond_bone = conditions[ 1 ], 
            ct_pil = ct_pil, 
            cbct_pil = cbct_pil )
        ct_tensor = rotated_images[ 0 ]
        cbct_tensor = rotated_images[ 1 ]
        
        # 之後需要可以再加 air_tensor 或 bone_tensor 
        return ct_tensor, cbct_tensor

    def __len__( self ) -> int:
        if self.mode == 'train':
            return self.boundary
        else:
            return self.dataset_len - self.boundary
    
    def _hu_condition( self, hu : np.ndarray ) -> List[ Image.Image ]:
        air_mask = thresh_mask( hu, thresh_val = -300, mode = 'bigger' )
        bone_mask = thresh_mask( hu, thresh_val = 200, mode = 'bigger' )
        body_mask = normalize( find_body( air_mask ) )

        cond_air = to_uint8_255( air_mask * body_mask )
        cond_bone = to_uint8_255( bone_mask * body_mask )

        return [ Image.fromarray( arr ).convert( "RGB" ).resize( ( 512, 512 ) ) for arr in [ cond_air, cond_bone ] ]
    
    def _preprocessing( 
        self,
        cond_air : Image.Image,
        cond_bone : Image.Image,
        ct_pil : Image.Image,
        cbct_pil : Image.Image ) -> List[ Image.Image ]:
        """
        _preprocessing:
        輸入一組 ct, cbct, air, bone 進行旋轉

        Args:
        ----------
        cond_air: Image.Image
        ----------
        cond_bone: Image.Image
        ----------
        ct_pil: Image.Image
        ----------
        cbct_pil: Image.Image

        Return:
        ----------
        List[ Image.Image ]
        """

        # 若當前模式為 train ，則進行旋轉的 augmentation
        if self.mode == 'train':
            # 先將各個影像轉換為單通道
            ct_pil = ct_pil.convert( mode = 'L' )
            cbct_pil = cbct_pil.convert( mode = 'L' )
            cond_air = cond_air.convert( mode = 'L' )
            cond_bone = cond_bone.convert( mode = 'L' )
            # 合併為一張 4 通道的 Image.Image
            ct_combined = Image.merge( mode = 'RGBA', bands = [ ct_pil, cbct_pil, cond_air, cond_bone ] )
            # 進行隨機旋轉
            ct_combined : Image.Image = image_augment_transforms( ct_combined )
            # 再將圖片分拆
            ct_pil, cbct_pil, cond_air, cond_bone = ct_combined.split()
        
        ct_tensor = image_transforms( img = ct_pil.convert( "RGB" ) )
        cbct_tensor = image_transforms( img = cbct_pil.convert( "RGB" ) )
        air_tensor = image_transforms( img = cond_air.convert( "RGB" ) )
        bone_tensor = image_transforms( img = cond_bone.convert( "RGB" ) )

        return [ ct_tensor, cbct_tensor, air_tensor, bone_tensor ]

"""
version 1:
    self.layerA = nn.Linear( in_features = io_size, out_features = latent_deg )
    self.layerB = nn.Linear( in_features = latent_deg, out_features = io_size )
"""

conv3x3 = lambda in_channels, out_channels, stride : nn.Conv2d( in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False )
conv1x1 = lambda in_channels, out_channels, stride : nn.Conv2d( in_channels, out_channels, kernel_size = 1, stride = stride, bias = False )

def show_history( data: Sequence[ Union[ int, float ] ], title: str, ylabel: str = None ) -> None:
    time = list( range( len( data ) ) )
    plt.figure()
    plt.plot( time, data, color = 'blue', linestyle = '-', marker = '' )
    plt.title( title )
    plt.xlabel( 'epoch' )
    if ylabel is not None:
        plt.ylabel( ylabel )
    # plt.show()
    @write_file
    def save_figure( file_path : str ):
        plt.savefig( file_path )

    save_figure( file_path = '{}/loss_history.png'.format( TUNING_SET_PREFIX ) )

    plt.close()
    
    return 

def vae_init( 
    model_name: str = pretrained_model_name_or_path,
    lora : bool = True 
    ) -> diffusers.AutoencoderKL:
    
    if lora:
        vae : diffusers.AutoencoderKL = AutoencoderKL.from_pretrained( model_name, subfolder = "vae", low_cpu_mem_usage = False )
    else:
        vae : diffusers.AutoencoderKL = diffusers.AutoencoderKL.from_pretrained( model_name, subfolder = "vae" )
    print( 'vae_init: pretrained ( fixed ) VAE loaded.' )
    return vae

def inference_pipeline( 
        image_idx   : int = 150,
        weight_name : str = '',
        epoch       : int = 1000,
        ts          : int = 20,
        lora_scale  : float = 1.0
         ):
    noise_scheduler : DDPMScheduler = DDPMScheduler.from_pretrained( pretrained_model_name_or_path = 'scheduler_config.json' )

    fixed_vae = vae_init( model_name = pretrained_model_name_or_path, lora = False )
    # fixed_vae.to( device = device, dtype = weight_dtype )
    fixed_vae.eval()
    print( 'inference: SD v1.5 vae loaded.' )

    vae = vae_init( model_name = pretrained_model_name_or_path, lora = True )
    vae.load_state_dict( state_dict = torch.load( 'lora_unit.pt' ) )
    # vae.to( device = device, dtype = weight_dtype )
    vae.eval()
    print( 'inference: pretrained vae with lora loaded.' )

    # weight_name = '0821_tuning/pretrained_weight/0823_epoch1000/align_unit_epoch{}'.format( epoch )
    weight_name = 'align_unit.pt'
    state_dict = torch.load( weight_name )

    print( 'inference: align unit loaded.' )

    cbct_image_paths = glob( '{}/*.png'.format( CBCT_DATASET_PATH ) )
    ct_image_paths = glob( '{}/*.png'.format( CT_DATASET_PATH ) )

    image_processor = VaeImageProcessor( vae_scale_factor = 8 )
    filtered_cbct = Image.open( cbct_image_paths[ image_idx ] ).resize( ( 512, 512 ) ).convert( "RGB" )
    filtered_ct = Image.open( ct_image_paths[ image_idx ] ).resize( ( 512, 512 ) ).convert( "RGB" )

    print( 'inference_pipeline: filename: {}'.format( cbct_image_paths[ image_idx ] ) )

    if ts is None:
        timesteps : torch.Tensor = torch.randint( low = 0, high = 1000, size = ( 1, ), device = device )
    else:
        timesteps : torch.Tensor = torch.tensor( data = [ ts ], device = device )
    timesteps = timesteps.long()
    print( 'timesteps: {}'.format( timesteps ) )

    def inference( 
        ct          : Image.Image,
        cbct        : Image.Image ) -> Tuple[ Image.Image, Image.Image, Image.Image, Image.Image ]:

        noise = torch.randn( size = ( 1, 4, 64, 64 ) ).to( device = device, dtype = weight_dtype )
        def run_encoder_and_add_noise( 
            encoder : Union[ diffusers.AutoencoderKL, AutoencoderKL ], 
            input: Image.Image ) -> torch.Tensor:
            input : torch.Tensor = image_transforms( input )

            input = torch.unsqueeze( input, dim = 0 )
            # shape: 1x3x512x512
            encoder.to( device = device, dtype = weight_dtype )

            # 這裡將 input tensor 透過 vae.encode() 成為一組 mean & var
            # => 格式：AutoencoderKLOutput, 繼承自 BaseOutput
            # 注意這裡直接再呼叫 latent_dist.sample() 進行重參數化
            # => 定義於 models/vae.py 中的 class DiagonalGaussianDistribution 中
            #   => 就是對給定的 mean 與 logvar 透過 randn_tensor

            # embedding 大小： 1x4x64x64
            if isinstance( encoder, AutoencoderKL ):
                embedding = encoder.encode( input.to( dtype = weight_dtype, device = device ), lora_scale = lora_scale ).latent_dist.sample()
            elif isinstance( encoder, diffusers.AutoencoderKL ):
                embedding = encoder.encode( input.to( dtype = weight_dtype, device = device ) ).latent_dist.sample()
            # noise = torch.randn_like( embedding )

            # 註：是否需要 scaling_factor ?
            # 可能需要：以免 tensor 內元素數值離 1 太遠
            # 基於 VQVAE 的離散特性，使用正規化破壞元素間的距離不是件好事
            # 2023-08-30: 經過 trace code 證實目前 stable diffusion 並不是使用 VQVAE ，而是回歸 VAE with KL-divergence 的老路
            # 2023-08-22: 實驗證實使用 scaling_factor 搭配 VaeImageProcessor 可以完成 enc-dec pipeline
            # 實作可參考 stable diffusion 系列 pipeline
            embedding = encoder.config.scaling_factor * embedding

            embedding = noise_scheduler.add_noise( embedding, noise, timesteps )
            encoder.to( device = 'cpu' )
            return embedding
        
        def run_decode( 
            encoder     : Union[ diffusers.AutoencoderKL, AutoencoderKL ], 
            input       : torch.Tensor ) -> torch.Tensor:
            
            out : torch.Tensor = encoder.decode( input / vae.config.scaling_factor, return_dict = False )[ 0 ]
            out = out.cpu().detach().to( dtype = torch.float32 )
            return out
        
        noisy_ct_embedding = run_encoder_and_add_noise( encoder = fixed_vae, input = ct ) # 加躁的 CT 
        noisy_cbct_embedding = run_encoder_and_add_noise( encoder = fixed_vae, input = cbct ) # 加躁的 CBCT
        noisy_cbct_lora_embedding = run_encoder_and_add_noise( encoder = vae, input = cbct ) # 加躁且過 LoRA 的 CBCT

        vae.to( device = device, dtype = weight_dtype )

        noisy_ct_image = run_decode( encoder = vae, input = noisy_ct_embedding ) # noisy_ct: 原始 encoder 的 CT 加躁
        noisy_cbct_image = run_decode( encoder = vae, input = noisy_cbct_embedding ) # noisy_cbct: 原始 encoder 的 CBCT 加躁
        noisy_cbct_lora_image = run_decode( encoder = vae, input = noisy_cbct_lora_embedding ) # noisy_cbct_lora: lora encoder 的 CBCT 加躁

        vae.to( device = 'cpu' )

        noisy_ct_result : List[ Image.Image ] = image_processor.postprocess( image = noisy_ct_image )
        noisy_cbct_result : List[ Image.Image ] = image_processor.postprocess( image = noisy_cbct_image )
        noisy_cbct_lora_result : List[ Image.Image ] = image_processor.postprocess( image = noisy_cbct_lora_image )
        return ( result[ 0 ], noisy_ct_result[ 0 ], noisy_cbct_result[ 0 ], noisy_cbct_lora_result[ 0 ] )
    
    result, ct_fixed, cbct_fixed, cbct_lora = inference( ct = filtered_ct, cbct = filtered_cbct )
    
    ct_fixed.convert( 'L' ).save( 'tuning-alignment-refct-{}-ts{}.png'.format( image_idx, ts ) )
    cbct_lora.convert( 'L' ).save( 'tuning-alignment-lora-{}-ts{}-scale{}.png'.format( image_idx, ts, lora_scale ) )
    result.convert( 'L' ).save( 'tuning-alignment-lora+align_unit-{}-ts{}.png'.format( image_idx, ts ) )
    cbct_fixed.convert( 'L' ).save( 'tuning-alignment-refcbct-{}-ts{}.png'.format( image_idx, ts ) )
    return 

def training_pipeline(  
        pretrained_encoder  : diffusers.AutoencoderKL,
        lora_encoder        : diffusers.AutoencoderKL,
        criterion           : Union[ nn.Module, Callable ],
        optimizer           : optim.Optimizer,
        training_dataset    : Dataset = None,
        epochs              : int = 400,
        batch_size          : int = 4,
        record_interval     : int = 100,
        training_type       : str = 'pickle',
    ) -> Dict[ str, Any ]:
    """
    training_pipeline:
    訓練一組 stable diffusion 的 vae-encoder
    這裡開啟 vae 中，attention processor 的 lora layer 作為實際在 tuning 的參數
    訓練目標是將輸入影像與 CT 的影像對齊
    可以認為是將 CBCT embedding 算得像 pretrained vae-encoder 對真實 CT 樣本產生的 embedding

    0915 實驗設定：
    這裡將輸入定義為 CBCT 的離散小波轉換 2 次分解低頻信號，也就是將第一次分解的低頻在丟去分解一次
    由於使用成對資料，因此可以讓被訓練的 encoder 知道預設的 encoder 在遇到真正 CT 時會如何表現

    Args:
    ----------
    pretrained_encoder: diffusers.AutoencoderKL
    stable diffusion 的 vae-encoder ，作為訓練對象的 ground truth
    ----------
    lora_encoder: AutoencoderKL
    基本與 stable diffusion vae-encoder 一樣，只差在 attention processor 有開啟 lora layer
    ----------
    criterion: Union[ nn.Module, Callable ]
    這裡採用的是 LA loss( 來自元淇的研究 )
    ----------
    optimizer: optim.Optimizere
    優化器
    ----------
    training_dataset: Dataset
    訓練資料集
    ----------
    epochs: int
    迭代次數
    ----------
    batch_size: int
    不解釋
    ----------
    record_interval: int
    每多少 epoch 存一次 checkpoint

    Return:
    ----------
    Dict[ str, Any ]
    最後的訓練結果
    """
    if training_dataset is None:
        training_dataset = AlignSet( boundary = 0.75, mode = 'train', source = training_type )
    
    training_loader = DataLoader( 
        dataset = training_dataset, 
        batch_size = batch_size,
        shuffle = True )

    # 初始化 loss buffer
    # ( 為了繪製 loss curve )
    loss_history: List[ float ] = []
    # 初始化進度條
    training_progress = tqdm( total = epochs, desc = 'training' )
    training_progress.clear()
    batch_bar = tqdm( total = len( training_loader ), desc = 'batch progress' )
    batch_bar.clear()
    # 載入 encoder 或 decoder
    pretrained_encoder.to( device, dtype = weight_dtype )
    pretrained_encoder.eval()
    lora_encoder.to( device = device, dtype = weight_dtype )
    lora_encoder.eval()
    lora_encoder.encoder.mid_block.attentions[ 0 ].processor.train() # 只開啟 lora processor 的 train()
    # 決定 noise scheduler
    # 這裡的選擇與加噪、降噪的 scale 有關
    # 由於 unet 的 pretrained weight 不變，這裡的選擇沒有到非常要緊
    noise_scheduler : DDIMScheduler = DDIMScheduler.from_pretrained( pretrained_model_name_or_path = 'vae_lora_test/scheduler_config.json' )
    # reconstruction error
    recon_criterion = nn.MSELoss()

    # 開始訓練囉
    for epoch in range( epochs ):
        total_loss = 0.0
        batch_bar.clear()
        batch_bar.reset()
        for ct_tensors, cbct_tensors in training_loader:
            # 從 loader 中取出 CT 與 CBCT
            ct_tensors: torch.Tensor = ct_tensors.to( device, dtype = weight_dtype )
            cbct_tensors: torch.Tensor = cbct_tensors.to( device, dtype = weight_dtype )

            # 清空優化器
            optimizer.zero_grad()
            
            # 算出 CT 通過原始 vae-encoder 後的 embedding: ct_latents
            with torch.no_grad():
                ct_latents: torch.Tensor = pretrained_encoder.encode( ct_tensors ).latent_dist.sample()
                ct_latents = pretrained_encoder.config.scaling_factor * ct_latents
            
            # 算出 CBCT 通過待訓練 lora-encoder 後的 embedding: cbct_latents
            # 注意這裡不是包在 torch.no_grad() 中
            cbct_latents: torch.Tensor = lora_encoder.encode( cbct_tensors ).latent_dist.sample()
            cbct_latents = lora_encoder.config.scaling_factor * cbct_latents

            # 選出 noise 與隨機 timesteps( 一個 batch 中每個 sample 有自己的 timestep )
            # 注意這裡 timesteps 會轉成長整數
            noise = torch.randn_like( cbct_latents )
            timesteps = torch.randint( low = 0, high = 1000, size = ( cbct_latents.shape[ 0 ], ), device = device )
            timesteps = timesteps.long()

            # 對 CT 與 CBCT 的 embedding 加噪
            noisy_ct_latents = noise_scheduler.add_noise( ct_latents, noise, timesteps )
            noisy_cbct_latents = noise_scheduler.add_noise( cbct_latents, noise, timesteps )
            
            # 初始化一個 loss
            loss : torch.Tensor = 0
            # 對 noisy_ct_latents 與 noisy_cbct_latents 的每個 channel 進行 LA 評估
            # loss 在這裡累加
            for cnt in range( noisy_ct_latents.shape[ 0 ] ):
                for ch in range( noisy_ct_latents.shape[ 1 ] ):
                    ct_mat = noisy_ct_latents[ cnt ][ ch ]
                    cbct_mat = noisy_cbct_latents[ cnt ][ ch ]

                    loss = loss + criterion( cbct_mat, ct_mat )
            
            # nan 與 loss 大小無關，已嘗試乘上 1e5
            # 後續實驗發現 nan 來自 torch.float16 導致的 overflow/underflow
            recon_loss : torch.Tensor = recon_criterion( noisy_cbct_latents, noisy_ct_latents )
            
            loss = loss + recon_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_bar.set_description( desc = 'epoch: {}, loss: {:.4f}'.format( epoch, loss.item() ) )
            batch_bar.update( 1 )

        eval_loss = total_loss / len( training_loader )
        loss_history.append( eval_loss )
        training_progress.update( 1 )
        
        if ( epoch + 1 ) % record_interval == 0:
            # torch.save( obj = model.state_dict(), f = '{}/align_unit_epoch{}.pt'.format( TUNING_SET_PREFIX, ( epoch + 1 ) ) )
            show_history( 
                data = loss_history, 
                title = 'loss in {} epochs'.format( epochs ), 
                ylabel = 'LA loss' )
        
        
    training_progress.close()
    batch_bar.close()
    show_history( 
        data = loss_history, 
        title = 'loss in {} epochs'.format( epochs ), 
        ylabel = 'LA loss' )
    
    return lora_encoder.state_dict()

def cal_CE_loss( 
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    reduction = 'none' ) -> torch.Tensor:
    """
    cal_CE_loss:
    計算 Cross Entropy 損失
    
    Args:
    ----------
    preds: torch.Tensor
    模型預測
    ----------
    targets: torch.Tensor
    答案
    ----------
    reduction: str, default: `none`
    計算模式，支援以下選擇：
    - `none`: 一般 CE loss
    - `mean`: 回傳 loss 的 mean

    Return:
    ----------
    torch.Tensor
    """
    log_softmax = nn.LogSoftmax( dim = -1 )
    loss : torch.Tensor = ( -targets * log_softmax( preds ) ).sum( 1 )
    
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()

def cal_LA_loss( 
    embedding1: torch.Tensor, # pred
    embedding2: torch.Tensor, # gt
    temperature: float = 1 ) -> torch.Tensor:
    """
    cal_LA_loss:
    計算 Latent Alignment 損失

    Args:
    ----------
    embedding1: torch.Tensor
    模型預測
    ----------
    embedding2: torch.Tensor
    答案
    ----------
    temperature: float, default: 1
    超參

    Return:
    ---------
    torch.Tensor
    """

    logits = ( embedding1 @ embedding2.T ) / temperature
    sim1 = embedding1 @ embedding1.T
    sim2 = embedding2 @ embedding2.T
    targets = F.softmax( ( sim1 + sim2 ) / 2*temperature, dim = -1 )
    loss1 = cal_CE_loss( preds = logits, targets = targets, reduction = 'none' )
    loss2 = cal_CE_loss( preds = logits.T, targets = targets.T, reduction = 'none' )
    return ( ( loss1 + loss2 )/2.0 ).mean()

def main( 
    vae           : diffusers.AutoencoderKL,
    mode            : str = 'eval',
    image_idx       : int = 150,
    epochs          : int = 5,
    flag_pretrained : bool = False,
    lora_scale      : float = 1.0,
    ts              : int = 200,
    pretrained_model_name_or_path : Optional[ str ] = 'runwayml/stable-diffusion-v1-5',
    weight_name     : Optional[ str ] = 'alignment_unit_default_name.pt',
    training_type   : Optional[ str ] = 'synthrad',
    ) -> None:
    torch.autograd.set_detect_anomaly( False )

    # 關閉 gradient 更新並載入 GPU
    vae.requires_grad_( False )
    vae.to( device, dtype = weight_dtype )
    print( 'device: {}'.format( device ) )

    if mode not in [ 'train', 'eval' ]:
        print( 'main: invalid pipeline mode: {}'.format( mode ) )
        print( 'set to `eval`' )
        mode = 'eval'
    
    if mode == 'eval':
        inference_pipeline( image_idx = image_idx, ts = ts, lora_scale = lora_scale )
    elif mode == 'train':
        batch_size = 4
        lora_encoder : diffusers.AutoencoderKL = AutoencoderKL.from_pretrained( 
            pretrained_model_name_or_path = pretrained_model_name_or_path, 
            subfolder = 'vae', 
            low_cpu_mem_usage = False )
        print( 'main: pretrained ( lora ) encoder loaded.' )
        
        if flag_pretrained is True:
            # print( 'main: pretrained weight loaded.' )
            pass
        # 應該是不能接兩個 generator 
        # 等等嘗試將兩個 generator 的結果合併成 list
        
        param_lora = [ param for param in lora_encoder.encoder.mid_block.attentions[ 0 ].processor.parameters() ]

        optimizer = optim.Adam( param_lora, lr = 1e-5 )
        alignment_weight = training_pipeline(  
            pretrained_encoder = vae,
            lora_encoder = lora_encoder,
            criterion = cal_LA_loss,
            optimizer = optimizer,
            batch_size = batch_size,
            epochs = epochs,
            training_type = training_type )
        
        if alignment_weight is None:
            return 
        torch.save( alignment_weight, weight_name )

    return 

def pickle_check( filename ):

    with open( filename, 'rb' ) as file:
        ct_dict : Dict[ str, Any ] = pickle.load( file )
        # name: str => dicom name
        # hu: np.ndarray => hu value of ct
        # cbct: Image.Image => sparse ver. of ct

        print( ct_dict.keys() )
        print( len( ct_dict[ 'conds' ] ) )
    return


if __name__ == '__main__':
    # pickle_check( filename = "0915_tuning/pickles/0.pkl")
    
    vae = vae_init( lora = True )
    arg_main_dict = {
        "vae" : vae,
        "mode" : 'train',
        "image_idx" : 151,
        "epochs" : 2,
        "flag_pretrained" : False,
        "lora_scale" : 1.0,
        "ts" : 500,
        "pretrained_model_name_or_path" : "stabilityai/stable-diffusion-2-base",
        "training_type" : "pickle",
    }
    main( **arg_main_dict )
    # weight_inspection()
    # unit_test()
    