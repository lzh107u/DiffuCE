from vae_lora_test import DBE
from cond_refine_decoder_utils import CRD
from cddm import CDDM

from glob import glob
from typing import List, Dict, Any, Union, Optional, Type, Tuple
from PIL import Image
from training_utils import (
    # load_image,
    dicom_preprocess,
    illumination_correction,
    figure_combine,
)
from dicom_utils import dicom_pipeline
import pickle
import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
import numpy as np
from math import log10
import json
from tqdm import tqdm
from skimage.metrics import structural_similarity, mean_squared_error # python -m pip install -U scikit-image
from sklearn.metrics import mean_absolute_error

from GANs.codes import GANBaseClass, gan_preprocessing
from torchvision import transforms
import re
import lpips # pip install lpips

IMG_FOLDER = '0915_tuning/pickles'

def load_pickle( file : str ) -> List[ Image.Image ]:
    
    with open( file, 'rb' ) as f:
        data_dict : Dict[ str, Union[ Image.Image, List[ torch.Tensor ], List[ Image.Image ], Dict[ str, float ] ] ] = pickle.load( f )
        ct : Image.Image = data_dict[ 'ct' ].convert( 'RGB' )
        cbct : Image.Image = data_dict[ 'cbct' ].convert( 'RGB' )
        cond : List[ Image.Image ] = data_dict[ 'conds' ]

        results = [ ct, cbct ]
        results.extend( cond )
    
        return results
    
def load_dicom( file : str ) -> Union[ List[ Image.Image ], None ]:
    """
    load_dicom:
    回傳 DiffuCE 需要的影像

    Args:
    ---------
    file: str
    dicom 檔位置

    Return:
    ---------
    Union[ List[ Image.Image ], None ]:
        ----- 
        List[ Image.Image ]:
        Dicom 檔內影像與 Hounsfield Unit 處理後影像
        [ ct/cbct, ct/cbct, air, bone, wavelet, tissue ]
        -----
        None:
        前處理錯誤
    """
    images = dicom_pipeline( ds_dir = file )
    if images is not None:
        ct = images[ 0 ]
        air = images[ 1 ]
        bone = images[ 2 ]
        wavelet = images[ 6 ]
        tissue = images[ 4 ]
        return [ ct, ct, air, bone, tissue, wavelet ] # CBCT dicom 不含 CT ground truth ，所以 0, 1 都是同一張
    else:
        return None
    
def load_gan_data( ds_dir : str, device : torch.device ) -> Tuple[ torch.Tensor, torch.Tensor, str ]:
    data : Tuple[ np.ndarray, np.ndarray, np.ndarray ] = gan_preprocessing( dicom = ds_dir )
    name = ds_dir.split( '/' )[ -1 ].split( '.' )
    name = '.'.join( name[ : len( name ) - 1 ] )
    img, air, bone = data
    img = torch.from_numpy( img )
        
    img = img.to( dtype = torch.float32, device = device )
    img = img.unsqueeze( dim = 0 )
    return img, air, name

def chest_set_parsing()->List[ str ]:
    """
    chest_set_parsing:
    回傳 chest CBCT validation set

    Args:
    ----------
    No

    Return:
    ----------
    List[ str ]
    回傳一個 List ，內部為 chest CBCT validation set 中全部的 dicom paths
    """
    img_folder = 'dataset/CBCT_img_folder_v2'
    chest_folders = glob( 'dataset/CBCT_img_folder_v2/00*' )
    chest_folders.sort()
    dicom_folders = glob( 'dataset/CHEST/*' )
    dicom_folders.sort()

    valid_dicom_paths = []
    
    
    for cnt, folder in enumerate( chest_folders ):
        if len( folder.split( '/' )[-1].split( '_' ) ) > 1:
            continue
        
        dicom_folder = dicom_folders[ cnt ]
        img_paths = glob( folder + '/*' )
        img_paths.sort()
        dicom_paths = glob( dicom_folder + '/CBCT/CT*.dcm')
        dicom_paths.sort()

        for img_path in img_paths:
            last_name = img_path.split('/')[-1].split('_')[-1].split('.')
            last_name = last_name[ : -1 ]
            last_name = '.'.join( last_name )
            # 用 / 切出最後的圖片檔名
            # 用 _ 分割最後檔名各片段
            # 用 . 分割副檔名

            valid_dicom_paths.append( dicom_folder + '/CBCT/' + last_name )
    
    valid_dicom_paths.sort()
    return valid_dicom_paths

def pair_eval( pred : Image.Image, gt : Image.Image, maxval : int = 255 ) -> Dict[ str, float ]:
    """
    pair_eval

    Return:
    ----------
    Dict[ str, float ]
    `mse`
    `mae`
    `psnr`
    `ssim`
    """
    pred = pred.convert( 'L' )
    gt = gt.convert( 'L' )

    pred_np = np.array( pred )
    gt_np = np.array( gt )

    if np.max( pred_np  ) == 0 or np.max( gt_np ) == 0:
        return { "exception" : True }

    ssim = structural_similarity( im1 = pred_np, im2 = gt_np )
    mse = mean_squared_error( image0 = pred_np, image1 = gt_np )

    
    mae = mean_absolute_error( y_true = gt_np, y_pred = pred_np )
    psnr = 10 * log10( maxval**2 / mse )

    results = {
        "mse" : mse,
        "mae" : mae,
        "psnr" : psnr,
        "ssim" : ssim,
        "exception" : False
    }

    return results 

class Evaluator:
    def __init__( self, device : torch.device = 'cpu' ) -> None:
        self.lpips_operator : nn.Module = lpips.LPIPS( net = 'alex' )
        self.img_processor : Type[ VaeImageProcessor ] = VaeImageProcessor( vae_scale_factor = 8 )
        self.device = device
        return
    
    def eval( 
        self, 
        pred : Image.Image, 
        gt : Image.Image, 
        maxval : int = 255,
        ) -> Dict[ str, float ]:

        dict_px = pair_eval( pred = pred, gt = gt, maxval = maxval )

        self.lpips_operator.to( self.device )

        pred_tensor : torch.Tensor = self.img_processor.preprocess( image = pred )
        gt_tensor : torch.Tensor = self.img_processor.preprocess( image = gt )

        pred_tensor = pred_tensor.to( self.device )
        gt_tensor = gt_tensor.to( self.device )

        lpips_score : torch.Tensor = self.lpips_operator( pred_tensor, gt_tensor )

        self.lpips_operator.to( 'cpu' )

        dict_px[ "lpips" ] = lpips_score.item()

        return dict_px

class DiffuCE:
    """
    DiffuCE:
    包含 DBE, CDM, CRD 三個模組
    無須擔心 GPU 用量問題，每一次使用都會自動 offload 
    """
    def __init__( 
        self,
        device : torch.device = 'cuda',
        dtype : torch.dtype = torch.float32,
        decoder_names : Optional[ List[ str ] ] = [ 'main' ],
        mode_names : Optional[ List[ str ] ] = [ 'air', 'bone', 'wavelet' ],
        cdd_state : Optional[ str ] = 'normal',
        dataset_mode : Optional[ str ] = 'default', # or `default`
        pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5',
        cdd_lora_ckpt_dir : Optional[ str ] = 'CDD_lora',
        decoder_cond_scale : Optional[ List[ float ] ] = [ 0.1, 0.05, 0.05 ],
        crd_version : Optional[ int ] = 8, 
    ) -> None:
        """
        __init__

        Args:
        ----------
        mode_names: Optional[ List[ str ] ] = [ `air`, `bone`, `wavelet` ]
        用於控制 latent control modules 的模式與順序
        ----------
        cdd_state: Optional[ str ]
        調整 latent U-Net 的樣式，有以下數種：
        1. `normal` : 正常模式，為帶有 ControlNet 與 LoRA 的 SD img2img pipeline
        2. `ablation` : Ablation Study 模式，不使用 ControlNet ，僅有 LoRA
        預設為 `normal`
        ----------
        decoder_names: Optional[ List[ str ] ] = [ `main` ]
        指示 decoder 的模式
        
        ----------
        pretrained_model_name_or_path: str = `runwayml/stable-diffusion-v1-5`
        base model 的版本，主要有以下兩種：
        `runwayml/stable-diffusion-v1-5` 
        `stabilityai/stable-diffusion-v2-base`
        ----------
        dataset_mode: Optional[ str ] = `default`
        資料來源，涉及前處理的模式，支援以下模式：
        `default`: 大林慈濟醫院資料集
        `synthrad2023`: MICCAI2023 SynthRAD task2 pelvis 資料集
        ----------
        cdd_lora_ckpt_dir: Optional[ str ] = 'generated_lora`
        CDD LoRA weight 位置
        ----------
        decoder_cond_scale: Optional[ List[ float ] ] = [ 0.1, 0.05, 0.05 ]
        各種條件在 Decoder 內的強度
        ----------
        crd_version: Conditional Refinement Decoder 的使用版本
        - 8: Dalin Hospital private set
        - 9: SynthRAD2023 task2(CBCT-to-CT) pelvis set


        Return:
        ----------
        None
        """
        if dataset_mode == 'default':
            ckpt_mode = 'default'
        elif dataset_mode == 'synthrad2023':
            ckpt_mode = 'synthrad'

        self.encoder = DBE( ckpt_mode = ckpt_mode )
        self.decoder = CRD( version = crd_version, mode_names = decoder_names )
        self.denoise_unit = CDDM( 
            mode_names = mode_names, 
            cdd_state = cdd_state, 
            dataset_mode = dataset_mode,
            pretrained_model_name_or_path = pretrained_model_name_or_path,
            lora_ckpt_dir = cdd_lora_ckpt_dir )
        self.device = device
        self.dtype = dtype
        self.timestep = 200
        self.scales_cond = decoder_cond_scale
        self.decoder_lora_scale = 0.1

    def param_setting(
        self,
        timestep : Optional[ int ] = None,
        scales_cond : Optional[ List[ float ] ] = None,
        decoder_lora_scale : Optional[ float ] = None,
    ) -> None:
        if isinstance( timestep, int ):
            self.timestep = timestep
        if isinstance( scales_cond, list ):
            self.scales_cond = scales_cond
        if isinstance( decoder_lora_scale, float ):
            self.decoder_lora_scale = decoder_lora_scale
    
    def __call__(
        self,
        cbct : Image.Image,
        cond : List[ Image.Image ],
        cdd_scales : Optional[ List[ float ] ] = [ 0.1, 0.05, 0.2, 0.2 ],
        encoder_main_state : Optional[ str ] = 'main',
        decoder_main_state : Optional[ str ] = 'main',
    ) -> Image.Image:
        """
        DiffuCE.__call__
        執行 DiffuCE framework 主程式

        Args:
        ----------
        cbct: Image.Image
        原始 CBCT 影像
        ----------
        cond: List[ Image.Image ]
        透過 Hounsfield Unit (HU) 算出的 air, bone 與二階小波轉換得來的 wavelet
        ----------
        cdd_scales: Optional[ List[ float ] ]
        用於 CDD 各 ControlNet module 的權重
        ----------
        encoder_main_state : Optional[ str ]
        調整 encoder 在對 input image 編碼時的模式，有以下數種：
        1. `main` : 正常模式，使用 alignment module 且依據 timestep 加噪
        2. `ablation` : Ablation Study 模式，不使用 alignment module 但加噪，即屬於預設的 SD encoder
        3. `cond` : 條件模式，不使用 alignment module 且不加噪，一般不選
        預設為 `main`
        ----------
        decoder_main_state : Optional[ str ]
        調整 decoder 在進行解碼時的行為
        1. `main` : 正常模式，所有 conditional guidance branches 將參與 main branch 的解碼
        2. `ablation` : Ablation Study 模式，不使用 conditional branch 與 main branch 的 LoRA ，視同一個 conditional guidance branch
        預設為 `main`


        Return:
        ----------
        Image.Image
        處理後 CT_recon 影像
        """
        if encoder_main_state not in [ 'main', 'cond', 'ablation' ]:
            print( "DiffuCE.__call__: invalid `encoder_main_state` : {}".format( encoder_main_state ) )
            encoder_main_state = 'main'
        if decoder_main_state not in [ 'main','ablation' ]:
            print( "DiffuCE.__call__: invalid `decoder_main_state` : {}".format( decoder_main_state ) )
            decoder_main_state = 'main'

        cbct_latent = self.encoder.encode( image = cbct, mode = encoder_main_state, timestep = self.timestep )
        air_latent = self.encoder.encode( image = cond[ 0 ], mode = 'cond' )
        bone_latent = self.encoder.encode( image = cond[ 1 ], mode = 'cond' )
        wavelet_latent = self.encoder.encode( image = cond[ 2 ], mode = 'cond' )
        cbct_ts0 = self.encoder.encode( image = cbct, mode = 'cond' )

        output = self.denoise_unit.denoise( image = cbct, cond = cond, latent = cbct_latent, scales = cdd_scales )

        """
        latent_dict = {
            "main" : output,
            "air" : air_latent,
            "bone" : bone_latent,
            "cbct" : cbct_ts0,
        }
        """
        # for synthrad2023
        latent_dict = { "main" : output, "wavelet" : wavelet_latent }


        if decoder_main_state == 'main':
            given_latent = latent_dict
            flag_control = False
        elif decoder_main_state == 'ablation':
            given_latent = output
            flag_control = True
        
        # flag_control 可以決定目前 decoder 的運算屬於 main branch 還是 conditional guidance branch
        # given_latent 是 Dict/torch.Tensor 也會對 decoder 的表現造成影響
        # 兩者可共同決定 decoder 的運作模式
        # 具體詳見 CRD 內部運作

        result : Image.Image = self.decoder.decode( 
            latent = given_latent, 
            scales_cond = self.scales_cond, 
            lora_scale = self.decoder_lora_scale,
            output_type = 'pil',
            flag_control = flag_control )
        
        result_corr = illumination_correction( image = result, mask = cond[ 0 ], output_type = 'pil' )

        return result_corr


def metric_analysis(
    filename : str,
):
    """
    metric_analysis:
    對 `metric_*_*.txt` 系列文件進行評估，並將結果顯示於螢幕上。

    Args:
    ----------
    filename: str
    檔名

    Return:
    ----------
    None
    """
    
    def parsing_func( text : str, buf : Dict[ str, Dict[ str, float ] ] ) -> Dict[ str, Dict[ str, float ] ]:
        """
        parsing_func:
        對每一組資料進行 parsing 與 summation

        Args:
        ----------
        text: str
        原始資料格式為 text
        ----------
        buf: Dict[ str, Dict[ str, float ] ]
        由 buffer_init() 得來的 buffer

        Return:
        ----------
        Dict[ str, Dict[ str, float ] ]
        更新後的 buffer
        """
        # raw data => dictionary
        top_layer_dict : Dict[ str, Union[ str, Dict[ str, float ] ] ] = json.loads( text )
        
        exp_types = buf.keys()
        # print( 'buf.keys: {}'.format( exp_types ) )
        # put new value in the right place in the buffer
        for exp in exp_types:
            try:
                exp_metrics = top_layer_dict[ exp ]
            except:
                continue
            exp_buffer : Dict[ str, float ] = buf[ exp ]
            running : Dict[ str, List ] = buf[ exp + '_rec' ]
            
            for metric in exp_buffer.keys():
                exp_buffer[ metric ] += exp_metrics[ metric ]
                running[ metric ].append( exp_metrics[ metric ] )
                
            buf[ exp ] = exp_buffer

        return buf
    
    def buffer_init( text : str ) -> Dict[ str, Dict[ str, float ] ]:
        """
        buffer_init:
        初始化一組 buffer 統計每一組樣本的表現，用於最終給予一個平均評價

        Args:
        ----------
        text : str
        輸入資料

        Return:
        ----------
        Dict[ str, Dict[ str, float ] ]
        用於統計的 buffer ，由數個 Dict 構成，各項指標以 0.0 初始化
        """
        # raw data => dictionary
        top_layer_dict : Dict[ str, Union[ str, Dict[ str, float ] ] ] = json.loads( text )
        top_names = top_layer_dict.keys()
        pattern = re.compile( "metric" )
        # initialize a buffer
        metric_overall = {}
        # find metrics with given pattern
        for name in top_names:
            match_result = pattern.search( name )
            if match_result is not None:
                # init 0.0 at every position in buffer
                buffer = { id : 0.0 for id in top_layer_dict[ name ] }
                metric_overall[ name ] = buffer
                record = { id : [] for id in top_layer_dict[ name ] }
                metric_overall[ name + '_rec' ] = record
        
        return metric_overall
    ### number of data in the document
    cnt = 0
    
    with open( filename, 'r' ) as f:
        line = f.readline().strip()

        buffer = buffer_init( text = line )
        while len( line ) > 0:
            
            buffer = parsing_func( text = line, buf = buffer )
            cnt += 1
            line = f.readline().strip()
            
    print( 'filename: {}'.format( filename ) )
    for name_layer1 in buffer:
        flag : bool = True
        for name_layer2 in buffer[ name_layer1 ]:
            try:
                buffer[ name_layer1 ][ name_layer2 ] /= cnt
            except:
                flag = False
                continue
            print( 'std {}, {} - {}'.format( name_layer1, name_layer2, np.std( buffer[ name_layer1 + '_rec' ][ name_layer2 ], ddof = 1 ) ) )
        if ( flag ):
            print( '{} : {}'.format( name_layer1, buffer[ name_layer1 ] ) )
        

    print( 'Hint: To evaluate the FID, use: `python -m pytorch_fid path/to/dataset1 path/to/dataset2`' )
    return 

def run_metric(
    model : Any = None,
    model_name : str = 'Default',
    data_mode : str = 'pickle',
    folder : Optional[ List[ str ] ] = None,
    date_suffix : str = '1215',
    eval_cnt : int = 500,
    flag_store : bool = False,
    flag_combine : bool = False,
    device : torch.device = 'cuda',
    fid_path : str = None,
    encoder_main_state : Optional[ str ] = 'main',
    decoder_main_state : Optional[ str ] = 'main',
    ):
    
    if data_mode not in [ 'pickle', 'dicom' ]:
        data_mode = 'dicom'
    
    if data_mode == 'pickle':
        files = glob( IMG_FOLDER + '/*.pkl' )
        
    elif data_mode == 'dicom':
        files = chest_set_parsing()
    
    if folder is not None:
        files = folder
    
    files.sort()

    print( 'DiffuCE, run_metric: There are {} files to evaluate.'.format( len( files ) ) )
    eval_operator = Evaluator( device = device )
    progress = tqdm( total = len( files ) )
    progress.clear()
    
    to_tensor_operator = transforms.ToTensor()
    for cnt, file in enumerate( files ):
        if cnt >= eval_cnt:
            # Early break
            break
        
        # 載入資料
        if data_mode == 'pickle':
            images : List[ Image.Image ] = load_pickle( file = file )
            name = 'default'
        elif data_mode == 'dicom':
            images : List[ Image.Image ] = load_dicom( file = file )
            name = file.split( '/' )[ -1 ].split( '.' )
            name = '.'.join( name[ : len( name ) - 1 ] )
        # 取資料時報錯
        if images is None:
            progress.update( 1 )
            continue

        ct = illumination_correction( images[ 0 ], mask = images[ 2 ] )
        cbct = illumination_correction( images[ 1 ], mask = images[ 2 ] )
        cond = images[ 2 : ]
        if model is None:
            cbct.save( fid_path + '/{}_{}.png'.format( model_name, cnt ) )
            progress.update( 1 )
            continue
        if isinstance( model, GANBaseClass ):
            # 將 PIL.Image 轉為 torch.Tensor
            cbct_tensor : torch.Tensor = to_tensor_operator( cbct.convert( 'L' ).resize( ( 512, 512 ) ) )
            # 拉伸到 -500 ~ +500
            cbct_tensor = cbct_tensor - torch.min( cbct_tensor )
            cbct_tensor = cbct_tensor / torch.max( cbct_tensor )
            cbct_tensor = cbct_tensor - 0.5
            cbct_tensor = cbct_tensor * 100
            cbct_tensor = cbct_tensor.to( device = device )

            result : Image.Image = model( cbct_tensor )
            result = illumination_correction( image = result, mask = cond[ 0 ] )
        elif isinstance( model, DiffuCE ):
            result : Image.Image = model( 
                cbct = cbct, 
                cond = cond, 
                encoder_main_state = encoder_main_state,
                decoder_main_state = decoder_main_state, ) 
            # 在 CDD.__call__ 內部會依據不同的 pipeline 進行調整，無須在外部調整輸入參數

        metric_baseline = eval_operator.eval( pred = cbct, gt = ct )
        metric_corr = eval_operator.eval( pred = result, gt = ct )
        
        with open( 'metric_{}_{}.txt'.format( model_name, date_suffix ), 'a+' ) as f:
            data_dict = {
                "index" : cnt,
                "path" : file,
                "metric_baseline" : metric_baseline,
                "metric_corr" : metric_corr,
            }
            json.dump( data_dict, f )
            f.write( '\n' )
            
            
        fig_imgs = {
            "CT" : ct,
            "CBCT" : cbct,
            "output" : result,
        }

        if flag_store is True:
            if flag_combine:
                figure_combine( 
                    images = fig_imgs, 
                    figname = 'dataset/folder1218/DiffuCE_{}_{}_{}_.png'.format( date_suffix, cnt, name ), 
                    plotname = 'DiffuCE_{}'.format( cnt ) )
            else:
                if fid_path is not None:
                    result.save( fid_path + '/{}_{}.png'.format( model_name, cnt ) )
                else:
                    result.save( '{}_{}.png'.format( model_name, cnt ) )
        progress.update( 1 )
    
    progress.close()
    if model is not None:
        metric_analysis( filename = 'metric_{}_{}.txt'.format( model_name, date_suffix ) )

    return 

def gen_blur_sample(
    img_idx : int,
) -> None:
    return 

def exp1227():
    model = DiffuCE( device = 'cuda', cdd_state = 'normal' ) # `normal` or `ablation`
    # model = None
    run_metric(
        model = model,
        model_name = 'DiffuCE',
        data_mode = 'pickle',
        folder = None,
        date_suffix = '0107',
        flag_store = True,
        flag_combine = False,
        device = 'cuda',
        fid_path = 'FID_dataset/DiffuCE_ablation_dec',
        encoder_main_state = 'main',
        decoder_main_state = 'ablation'
    )

def metric_eval( 
    filename : str = 'metric_DiffuCE_1113.txt', 
    col_names : List[ str ] = [ 'metric_cbct', 'metric_result', 'metric_corr' ]
):
    def buffer_init( length : int, val : float = 0.0 ) -> List[ float ]:
        buffer = [ val for i in range( length ) ]
        return buffer
    
    with open( filename, 'r' ) as f:
        line = f.readline().strip()
        total = 1
        data_dict : Dict[ str, Union[ str, Dict[ str, float ] ] ] = json.loads( line )

        class_names = data_dict.keys()
        for name in class_names:
            if isinstance( data_dict[ name ], dict ):
                break
        # print( 'class_names:', class_names )
        sample_dict = data_dict[ name ]
        col_names = sample_dict.keys()

        running_metrics = {}
        for name in col_names:
            buffer = buffer_init( length = len( class_names ), val = 0.0 )
            running_metrics[ name ] = buffer

        best_psnr = 0.0
        pos = None
        best_type = None
        print( 'running keys: ', running_metrics.keys() )
        while line:
            data_dict : Dict[ str, Union[ str, Dict[ str, float ] ] ] = json.loads( line )

            for cnt_cls, cls_name in enumerate( class_names ):
                if not isinstance( data_dict[ cls_name ], dict ):
                    continue

                metric_dict = data_dict[ cls_name ]
                for cnt_col, col_name in enumerate( col_names ):
                    # print( 'rm-keys: {}'.format( running_metrics.keys() ) )
                    running_metrics[ col_name ][ cnt_cls ] += metric_dict[ col_name ]

                    if col_name == 'psnr' and metric_dict[ col_name ] > best_psnr:
                        best_psnr = metric_dict[ col_name ]
                        pos = data_dict[ 'index' ]
                        best_type = cnt_cls
            line = f.readline().strip()
            total += 1

    for col_name in col_names:
        
        print( '---------------------- Metric: {} -----------------'.format( col_name ) )
        for cnt, cls_name in enumerate( class_names ):
            if col_name not in running_metrics.keys():
                continue
            
            print( '{} : {}'.format( cls_name, running_metrics[ col_name ][ cnt ]/total ) )
    
    print( '--------------------------------------------------------------' )
    print( 'best psnr: {}, pos: {}, type: {}'.format( best_psnr, pos, best_type ) )
    
    return 

def inference( file_idx : int = 0, tissue_strength : float = 0.3 ):

    framework = DiffuCE( mode_names = [ 'air', 'bone', 'tissue', 'wavelet' ] )
    files = chest_set_parsing()
    files.sort()
    images = load_dicom( file = files[ file_idx ] )
    cdd_scales = [ 0.1, 0.1, tissue_strength, 0.3 ]
    result = framework( cbct = images[ 0 ], cond = images[ 2 : ], cdd_scales = cdd_scales )

    images[0].save( 'Default_input.png' )
    result.save( 'controlnet_1228_tissue_hu_ChestSet{}_004.png'.format( file_idx ) )

    return 

if __name__ == '__main__':
    # print( 'DiffuCE starts.' )
    # 訓練集：'0915_tuning/pickles'
    # eval_set = '0915_tuning/pickles/*.pkl'
    # 外部驗證集：'dataset/ABD/*/CBCT/CT*.dcm'
    # dicoms = chest_set_parsing()
    
    # run_metric( folder = None, date_suffix = '0521', eval_cnt = 500, image_store = False, data_mode = 'pickle' )
    pass
    # exp1227()
    inference()
    # metric_analysis( filename = 'metric_doc/metric_DiffuCE_ablation_dec_0107.txt' )