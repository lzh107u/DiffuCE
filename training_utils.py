from typing import Sequence, Union, Optional, List, Callable, Any
from matplotlib import pyplot as plt
from math import floor, ceil, log, pow
import numpy as np
import cv2
from PIL import Image as Image
from typing import Dict
from glob import glob
import pickle
from skimage._shared.utils import check_shape_equality, _supported_float_type, warn
import functools
from skimage._shared import utils
from skimage.util.arraycrop import crop
from skimage.util.dtype import dtype_range
import os
from functools import wraps
from datetime import datetime
import warnings
from inspect import signature

from dicom_utils import ( 
    dicom_pipeline, 
    normalize, 
    to_uint8_255, 
    thresh_mask, 
    gen_tissue, 
    gen_sobel, 
    single_wavelet_2d, 
    wavelet_recon )

CBCT_FOLDER = 'dataset/ABD/*/CBCT/CT*.dcm'
CT_FOLDER = 'dataset/ABD/*/CT/CT*.dcm'
PICKLE_FOLDER = '0915_tuning/CBCT'

__all__ = [
    'dicom_proprocess',
    'wavelet_process',
    'load_image',
    'show_history',
    'figure_combine',
    'ch_transform',
    'pixelwise_accessment'
    'illumination_correction',
]

def dicom_preprocess( 
    cond_mode   : List[ str ], 
    dicom_idx   : int = 0, 
    data_mode   : str = 'CBCT',
    given_path  : str = '',
    flag_store  : Optional[ bool ] = False,
    check_dir   : str = None,
     ) -> Union[ List[ Image.Image ], None ]:
    """
    dicom_preprocess:
    透過給予的 idx 與 mode ，開啟相應的 dicom 並回傳對應的 condition
    若給的是特定檔案的位址也可以

    Args:
    ----------
    mode: str
    condition 模式
    ----------
    dicom_idx: int
    dicom 檔的索引，
    這裡直接對所有 folder 中的 dicom 檔進行索引
    ----------
    given_path: str
    給予的指定位置，預設為 None

    Return:
    ----------
    List[ Image.Image ]
    原圖與 condition 
    以上都已經進行過前處理，可直接送入模型中
    0. CBCT
    1. None | CT , 這裡存放原始 CT ，若不存在就是 None
    2. ~ Conditions

    """
    if data_mode == 'CBCT':
        folder = CBCT_FOLDER
    else:
        folder = CT_FOLDER
    
    dicom_names = glob( folder )
    # folder options:
    # => dicom_folder
    # => cbct_folder
    if check_dir is None and len( given_path ) == 0:
        dicom_path = dicom_names[ dicom_idx ]
    elif len( given_path ) > 0:
        dicom_path = given_path
    else:
        dicom_path = check_dir

    # print('dicom file name: {}'.format( dicom_path ) )
    images: Sequence[ Image.Image ] = dicom_pipeline( dicom_path )
    if images is None:
        return None
    
    conditions : List[ Image.Image ] = [ images[ 0 ], None ]

    for mode in cond_mode:
        if mode == 'air':
            conditions.append( images[ 1 ] )
        elif mode == 'bone':
            conditions.append( images[ 2 ] )
        elif mode == 'tissue':
            conditions.append( images[ 4 ] )
        elif mode == 'body':
            conditions.append( images[ 5 ] )
        elif mode == 'wavelet':
            conditions.append( images[ 6 ] )
    if flag_store is True:
        images[ 0 ].save( 'controlnet-la-idx{}-CT.png'.format( dicom_idx ) )
    return conditions

def wavelet_process( image : Image.Image ) -> Image.Image:

    LL, HH = single_wavelet_2d( image.convert( 'L' ) )
    LL, HH = single_wavelet_2d( LL )
    cbct_wavelet_lv2 = Image.fromarray( to_uint8_255( LL ) ).resize( ( 512, 512 ) )

    return cbct_wavelet_lv2

def load_image( 
    idx         : int = 200,
    cond_mode   : List[ str ] = [ 'bone' ],
    flag_store  : Optional[ bool ] = True,
    path        : Optional[ str ] = None,
    flag_wavelet: Optional[ bool ] = False,
    ) -> List[ Image.Image ]:
    """
    load_image:
    讀取影像與其附加資料，包含以下幾項資料：
    - filtered_ct( 已濾除床鋪 )
    - hu

    Args:
    ----------
    idx: int
    指定要讀取的資料，取值範圍：[ 0, 399 ]
    ----------
    cond_mode: List[ str ]
    指定要回傳的條件影像
    目前有以下模式：
    - air
    - bone
    - tissue
    - wavelet
    - body

    Return:
    ----------
    List[ Image.Image ]
    回傳 unet 與 controlnet 的輸入影像
    0. 待輸入之 CBCT
    1. 原始 CT
    2. ~ ControlNets 輸入

    """
    post_proc : Callable[ [ np.ndarray ], Image.Image ] = lambda arr : Image.fromarray( arr ).convert( "RGB" ).resize( ( 512, 512 ) )
    
    if path is None:
        path = '{}/ct{}.pkl'.format( PICKLE_FOLDER, idx )


    with open( path, 'rb' ) as f:
        data_dict : Dict[ str, Any ] = pickle.load( f )
        # filtered_ct: Image.Image, RGB, 512x512
        # name: str
        # hu: np.ndarray
        # cbct: Image.Image, RGB, 512x512
        # body_mask: Image.Image, RGB, 512x512
        if 'ct' in data_dict.keys():
            ct_key = 'ct'
        elif 'filtered_ct' in data_dict.keys():
            ct_key = 'filtered_ct'
        else:
            raise ValueError( 'No ct exist in pickle file.' )
        filtered_ct : Image.Image = data_dict[ ct_key ]
        hu : np.ndarray = data_dict[ 'hu' ]
        body_mask : Image.Image = data_dict[ 'body_mask' ]
        cbct : Image.Image = data_dict[ 'cbct' ]

    air_mask = thresh_mask( hu, thresh_val = -300, mode = 'bigger' )
    bone_mask = thresh_mask( hu, thresh_val = 200, mode = 'bigger' )

    body_np = np.array( body_mask.convert( 'L' ) )
    
    cond_air = to_uint8_255( air_mask * body_np )

    filtered_ct_np = to_uint8_255( np.array( filtered_ct.convert( 'L' ) ) )
    sobel = gen_sobel( filtered_ct_np )
    inv_bone = thresh_mask( hu, thresh_val = 200, mode = 'smaller' )
    sobel = sobel * inv_bone * air_mask
    sobel = thresh_mask( image = sobel, thresh_val = 0.05, mode = 'bigger' )
    cond_sobel = to_uint8_255( sobel )

    cond_bone = to_uint8_255( bone_mask * body_np )

    cond_tissue = to_uint8_255( 
        gen_tissue( 
            q_factor = 5, 
            c_factor = 3, 
            target_ct = filtered_ct_np, 
            air_mask = air_mask*body_np, 
            bone_mask = bone_mask * body_np, 
            flag_bone = True,
            top_k = 5 ) )

    cond_wavelet, hh = single_wavelet_2d( image = filtered_ct.convert( 'L' ) )
    if len( cond_wavelet.shape ) >= 3:
        raise ValueError( 'ct-dataset-0905: invalid wavelet condition' )
    cond_wavelet, hh = single_wavelet_2d( image = cond_wavelet )
    cond_wavelet = to_uint8_255( cond_wavelet )

    if flag_wavelet is True:
        cbct = wavelet_process( cbct )

    conditions : List[ Image.Image ] = [ cbct, filtered_ct ]
    for mode in cond_mode:
        if mode == 'air':
            conditions.append( post_proc( cond_air ) )
        elif mode == 'bone':
            conditions.append( post_proc( cond_bone ) )
        elif mode == 'tissue':
            conditions.append( post_proc( cond_tissue ) )
        elif mode == 'body':
            conditions.append( body_mask )
        elif mode == 'wavelet':
            conditions.append( post_proc( cond_wavelet ) )
    if flag_store is True:
        filtered_ct.save( 'controlnet-la-idx{}-CT.png'.format( idx ) )
    # 回傳資料
    return conditions

def show_history( 
    data: Union[ Sequence[ Union[ int, float ] ], Dict[ str, Sequence[ Union[ int, float ] ] ] ], 
    title: str, 
    ylabel: str = None,
    folder_prefix: Optional[ str ] = None,
    filename: Optional[ str ] = 'loss_history'
    ) -> None:
    """
    show_history:
    輸入給定的紀錄，輸出一個折線圖。
    用 list 存每個 epoch 的紀錄( loss or metric )並丟進來就行

    Args:
    ----------
    data: Sequence[ Union[ int, float ] ]
    紀錄，像是 loss 或 metric ，建議以 list 的格式傳入
    ----------
    title: str
    圖的標題
    ----------
    ylabel: str
    標記 y 座標為何種數值
    注意： x 座標都是 epoch
    ----------
    folder_prefix: str
    待儲存的 folder
    ---------
    filename: str
    輸出檔案的名稱，副檔名已寫死為 .png

    Return:
    ---------
    None
    
    """
    
    plt.figure()
    if not isinstance( data, dict ):
        time = list( range( len( data ) ) )
        plt.plot( time, data, color = 'blue', linestyle = '-', marker = '' )
    else:
        for name in data:
            time = list( range( len( data[ name ] ) ) )
            plt.plot( time, data[ name ], linestyle = '-', marker = '', label = name )
        plt.legend()
    plt.title( title )
    plt.xlabel( 'epoch' )
    if ylabel is not None:
        plt.ylabel( ylabel )
    # plt.show()
    
    @write_file
    def save_figure( file_path: str ):
        plt.savefig( file_path )

    if folder_prefix is not None and isinstance( folder_prefix, str ):
        # plt.savefig( '{}/{}.png'.format( folder_prefix, filename ) )
        save_figure( '{}/{}.png'.format( folder_prefix, filename ) )
    else:
        # plt.savefig( '{}.png'.format( filename ) )
        save_figure( '{}.png'.format( filename ) )

    plt.close()
    
    return 

def figure_combine(
    images      : Dict[ str, Image.Image ],
    figname     : str,
    plotname    : str,
    ) -> None:
    """
    figure_combine:
    將傳入的影像全部合併為一張大圖

    Args:
    ----------
    images: Dict[ str, Image.Image ]
    所有待合併影像，合併後大圖會顯示每一張圖的 name 
    ----------
    figname: str
    大圖檔名
    ----------
    plotname: str
    大圖標題

    Return:
    ---------
    None
    
    """
    
    width = 4
    height = ceil( len( images )/width )
    background = np.zeros( ( 512, 512 ), dtype = np.uint8 )
    figsize = ( width + 2, height + 2 )
    
    fig, axs = plt.subplots( height, width, figsize = figsize )

    fig.suptitle( plotname )
    names = list( images.keys() )
    
    if len( images ) != width * height:
        print( 'training_utils, figure_combine: the number of `images` doesn\'t match the figsize {}'.format( ( width, height ) ) )
    
    for idx in range( height * width ):
        if idx >= len( images ):
            image = background
            label = 'empty'
        else:
            image = images[ names[ idx ] ]
            image = np.array( image.convert( 'L' ) )
            label = names[ idx ]

        try:
            axs[ floor( idx / width ) ][ idx % width ].imshow( image, cmap = 'gray' )
            axs[ floor( idx / width ) ][ idx % width ].set_title( label, fontsize = 6 )
            axs[ floor( idx / width ) ][ idx % width ].set_axis_off()
        except TypeError:
            axs[ idx % width ].imshow( image, cmap = 'gray' )
            axs[ idx % width ].set_title( label, fontsize = 6 )
            axs[ idx % width ].set_axis_off()

    plt.tight_layout()

    @write_file
    def save_figure( file_path: str ):
        fig.savefig( file_path, format = 'png', dpi = 512, bbox_inches = 'tight' )
    # fig.savefig( figname, format = 'png', dpi = 512, bbox_inches = 'tight' )
    save_figure( figname )
    plt.close()
    
    return 

def ch_transform(
    input : np.ndarray,
    mode : str = '3to1', # or `1to3`
    ) -> np.ndarray:

    if mode == '3to1':
        maxval = np.max( input )
        if maxval > 1:
            input = input.astype( np.uint8 )
        input = cv2.cvtColor( input, cv2.COLOR_RGB2GRAY )
    
    elif mode == '1to3':
        maxval = np.max( input )

        if maxval <= 1:
            input = input * 255
            input = input.astype( np.uint8 )
        
        input = cv2.cvtColor( input, cv2.COLOR_GRAY2RGB )

    return input


def pixelwise_accessment(
    pred            : Union[ Image.Image, np.ndarray ], 
    gt              : Union[ Image.Image, np.ndarray ],
    threshold       : float = 0.5,
    designated_val  : Optional[ float ] = None,
    output_type     : str = 'pil', # or np
    flag_abs        : bool = True, 
) -> Union[ Image.Image, np.ndarray ]:
    """
    depricated to `pixelwise_assessment`
    """
    return pixelwise_assessment( pred, gt, threshold, designated_val, output_type, flag_abs )

def pixelwise_assessment( 
    pred            : Union[ Image.Image, np.ndarray ], 
    gt              : Union[ Image.Image, np.ndarray ],
    threshold       : float = 0.5,
    designated_val  : Optional[ float ] = None,
    output_type     : str = 'pil', # or np
    flag_abs        : bool = True, 
    ) -> Union[ Image.Image, np.ndarray ]:
    """
    pixelwise_accessment:
    對 pred 與 gt 進行像素級別的比較，超過一定閾值的位置就標記為紅色

    Args:
    ---------
    pred: Union[ Image.Image, np.ndarray ]
    模型預測結果
    ---------
    gt: Union[ Image.Image, np.ndarray ]
    比較基準
    ---------
    threshold: float
    比較閾值，會將殘差正規化到 0-1 之間
    ---------
    designated_val: Optional[ float ]
    不對殘差進行正規化，以 designated_val 作為閾值
    ---------
    output_type: str
    輸出類型，目前支援以下模式：
    `pil`: Image.Image
    `np`: np.ndarray
    ----------
    flag_abs: bool
    是否對殘差進行絕對值計算，預設開啟

    Return:
    ----------
    Union[ Image.Image, np.ndarray ]
    回傳結果，這裡對 R 通道進行混和，用以凸顯差異較大的位置
    資料類別由 output_type 決定

    """
    def type_check( array : Union[ np.ndarray, Image.Image ] ) -> np.ndarray:
        # 確認型別與通道數
        if not isinstance( array, np.ndarray ):
            try:
                array = array.convert( 'L' )
            except:
                raise TypeError( 'Data type of given image is invalid. Require `Image.Image` or `np.array`' )
            array = np.array( array )
        if len( array.shape ) >= 3:
            array = ch_transform( input = array, mode = '3to1' )
        return array
    # 確定 input 都沒問題
    pred = type_check( pred )
    gt = type_check( gt )
    # 大小不同就報錯
    if pred.shape != gt.shape:
        raise ValueError( 'The shape of `pred` and `gt` should be the same' )
    # 計算差值
    pred = np.int16( pred )
    gt = np.int16( gt )
    residual = pred - gt
    if flag_abs is True:
        # 這裡對差值計算絕對值，由 flag_abs 控制，預設開啟
        residual = np.abs( residual )
    maxval = np.max( residual )
    if designated_val is None or designated_val >= maxval:
        # 判斷 designated_val 是否介於 residual 值域中，若否則使用 threshold
        designated_val = threshold
        residual = residual - np.min( residual )
        residual = residual / np.max( residual )

    map_exceed = np.where( residual > designated_val, 1, 0 )
    
    pred = np.uint8( pred )
    pred = ch_transform( pred, '1to3' )

    def image_fusion( array: np.ndarray, mask: np.ndarray ) -> np.ndarray:
        # R 通道用於顯示差值結果
        array[ : , : , 0 ] = np.uint8( mask * 255 )
        # G, B 通道不變
        array[ : , : , 1 ] *= np.uint8( mask )
        array[ : , : , 2 ] *= np.uint8( mask )
        return array
    # 合併比較結果
    pred = image_fusion( array = pred, mask = map_exceed )
    # 依照指定的資料類別回傳
    if output_type == 'pil':
        pred = Image.fromarray( pred )
        return pred.convert( 'RGB' )
    elif output_type == 'np':
        return pred
    
def illumination_correction(
    image       : Union[ Image.Image, np.ndarray ],
    mask        : Union[ Image.Image, np.ndarray ],
    mean_std    : int = 120,
    output_type : Optional[ str ] = 'pil',
    flag_hist : Optional[ bool ] = False,
    ) -> Union[ Image.Image, np.ndarray ]:
    """
    illumination_correction:
    將輸入影像透過 gamma correction 校正到給定的範圍上，
    這裡 gamma 的計算方式是透過轉換前後的均值關係計算而來

    Args:
    ---------
    image: Union[ Image.Image, np.ndarray ]
    輸入影像，僅限單通道( 三通道 PIL 會進行轉換 )
    ---------
    mask: Union[ Image.Image, np.ndarray ]
    用於遮住背景的遮罩，傳入來自 dicom_pipeline 的 air_mask 即可。
    不使用 mask 會導致校正時大量背景值汙染轉換參數
    ---------
    mean_std: int
    轉換後均值，預設為 120
    ----------
    output_type: Optional[ str ]
    輸出資料型態，預設為 `pil`

    Return:
    ---------
    Union[ Image.Image, np.ndarray ]
    輸出轉換亮度後的影像，預設為三通道 PIL 影像
    """
    # 確認輸入無異常
    if not isinstance( output_type, str ) or output_type not in [ 'pil', 'np' ]:
        print( 'illumination-correction: invalid `output_type` is given, set to default: `pil`.' )
        output_type = 'pil'
    # mask.save('prep-mask-bug.png')
    if not isinstance( image, np.ndarray ):
        image = np.array( image.convert( 'L' ) )
    if not isinstance( mask, np.ndarray ):
        mask = np.array( mask.convert( 'L' ) )
    
    
    if ( np.min( mask ) == np.max( mask ) ) :
        # print('illumination_correction: invalid min-max range {} ~ {}'.format( np.min( mask ), np.max( mask ) ) )
        return image
    # 對非背景區進行統計
    data = []
    mask = mask - np.min( mask )
    mask = mask / np.max( mask )

    for row in range( image.shape[ 0 ] ):
        for col in range( image.shape[ 1 ] ):
            if mask[ row, col ] > 0:
                # 只收集大於 0 的位置
                data.append( image[ row, col ] )
    
    hist, bins = np.histogram( data, bins = 256, range = [ 0, 256 ] )
    pos_min = 0
    pos_max = 0
    for i in range( len( hist ) ):
        if ( hist[ i ] >= 1000 and pos_min == 0 and i > 30 ):
            # i > 30 是經驗設定，當 pixel value 小於 30 時，多半不是有效數據
            pos_min = i
            break
    for i in range( len( hist ) ):
        if ( hist[ len( hist ) - i - 1 ] >= 1000 ):
            pos_max = len( hist ) - 1 - i
            break
    # hist: 各個 bin 上的數量，這裡設 256 ，可以直接視為各 pixel value 的數量
    # bins: 每個 bin 各自代表的數值，可以當作是每個 bin 自己的值域
    try:
        cdf = hist[ pos_min : pos_max ].cumsum()
        cdf = ( cdf - cdf.min() ) / ( cdf.max() - cdf.min() )
        cdf = cdf * ( pos_max - pos_min ) + pos_min
    except:
        if output_type == 'np':
            return image
        # 轉換至 PIL 影像，且為三通道
        image = Image.fromarray( image )
        image = image.convert( 'RGB' )

        return image
    
    # 進行 gamma 轉換
    mean_in = 0
    total_cnt = 0
    for val, cnt in enumerate( hist ):
        mean_in += val*cnt
        total_cnt += cnt
    # 由次數 * 像素值再除以總數得到平均值( 期望值 )
    mean_in = mean_in / total_cnt

    # gamma transform: y = x^r
    # x: 輸入，mean_in
    # y: 輸出，mean_std
    # r: 校正參數，這裡透過兩側開 log 並移項計算 r( gamma ) 的具體數值 
    # log(y) = r * log(x), r = log(y) / log(x)
    gamma = log( mean_std ) / log( mean_in )

    # 為了避免 pixel value 數值溢出，這裡轉換為 uint16
    image = np.uint16( image )
    # 對每個 pixel 進行 gamma 轉換
    
    for row in range( image.shape[ 0 ] ):
        for col in range( image.shape[ 1 ] ):
            if image[ row, col ] >= pos_min and image[ row, col ] < pos_max and flag_hist is True:
                image[ row, col ] = cdf[ image[ row, col ] - pos_min ]
                
            if mask[ row, col ] > 0 and image[ row, col ] > 30:
                
                newval = floor( pow( image[ row, col ], gamma ) )
                if newval > 255:
                    newval = 255
                image[ row, col ] = newval
    
    image = np.uint8( image )

    # 若 output_type 為 `np`，回傳單通道 np.uint8 影像
    if output_type == 'np':
        return image
    
    # 轉換至 PIL 影像，且為三通道
    image = Image.fromarray( image )
    image = image.convert( 'RGB' )

    return image

def write_file( func ):
    """
    寫檔用的修飾子，會自動判定寫入位置是否存在，若不存在會創造一個新的
    """
    @wraps( func )
    def wrapper( *args, **kwargs ):
        # 這裡預設檔案路徑是作為第 0 個參數傳入
        file_path = args[ 0 ] if args else kwargs.get( 'file_path' )
        if not isinstance( file_path, str ):
            raise ValueError("The name of path argument is stricted to `file_path`; Please check the correctness of file writing function.")

        if file_path:
            directory = os.path.dirname( file_path )
            if len(directory) > 0 and not os.path.exists( directory ):
                os.makedirs( directory )
        
        base, ext = os.path.splitext( file_path )
        counter = 1
        new_file_path = file_path
        while os.path.exists(new_file_path):
            new_file_path = f"{base}_{counter}{ext}"
            counter += 1
        
        if 'file_path' in kwargs:
            kwargs[ 'file_path' ] = new_file_path
        else:
            args = list( args )
            args[ 0 ] = new_file_path
            args = tuple( args )

        return func( *args, **kwargs )
    return wrapper



def structural_similarity(
    im1,
    im2,
    *,
    win_size=None,
    gradient=False,
    data_range=None,
    channel_axis=None,
    gaussian_weights=False,
    full=False,
    **kwargs,
):
    """
    Compute the mean structural similarity index between two images.
    Please pay attention to the `data_range` parameter with floating-point images.

    Parameters
    ----------
    im1, im2 : ndarray
        Images. Any dimensionality with same shape.
    win_size : int or None, optional
        The side-length of the sliding window used in comparison. Must be an
        odd value. If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.
    gradient : bool, optional
        If True, also return the gradient with respect to im2.
    data_range : float, optional
        The data range of the input image (difference between maximum and
        minimum possible values). By default, this is estimated from the image
        data type. This estimate may be wrong for floating-point image data.
        Therefore it is recommended to always pass this scalar value explicitly
        (see note below).
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    full : bool, optional
        If True, also return the full structural similarity image.

    Other Parameters
    ----------------
    use_sample_covariance : bool
        If True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        Algorithm parameter, K1 (small constant, see [1]_).
    K2 : float
        Algorithm parameter, K2 (small constant, see [1]_).
    sigma : float
        Standard deviation for the Gaussian when `gaussian_weights` is True.

    Returns
    -------
    mssim : float
        The mean structural similarity index over the image.
    grad : ndarray
        The gradient of the structural similarity between im1 and im2 [2]_.
        This is only returned if `gradient` is set to True.
    S : ndarray
        The full SSIM image.  This is only returned if `full` is set to True.

    Notes
    -----
    If `data_range` is not specified, the range is automatically guessed
    based on the image data type. However for floating-point image data, this
    estimate yields a result double the value of the desired range, as the
    `dtype_range` in `skimage.util.dtype.py` has defined intervals from -1 to
    +1. This yields an estimate of 2, instead of 1, which is most often
    required when working with image data (as negative light intensities are
    nonsensical). In case of working with YCbCr-like color data, note that
    these ranges are different per channel (Cb and Cr have double the range
    of Y), so one cannot calculate a channel-averaged SSIM with a single call
    to this function, as identical ranges are assumed for each channel.

    To match the implementation of Wang et al. [1]_, set `gaussian_weights`
    to True, `sigma` to 1.5, `use_sample_covariance` to False, and
    specify the `data_range` argument.

    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_ssim`` to
        ``skimage.metrics.structural_similarity``.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       :DOI:`10.1109/TIP.2003.819861`

    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       :arxiv:`0901.0065`
       :DOI:`10.1007/s10043-009-0119-z`

    """
    check_shape_equality(im1, im2)
    float_type = _supported_float_type(im1.dtype)

    if channel_axis is not None:
        # loop over channels
        args = dict(
            win_size=win_size,
            gradient=gradient,
            data_range=data_range,
            channel_axis=None,
            gaussian_weights=gaussian_weights,
            full=full,
        )
        args.update(kwargs)
        nch = im1.shape[channel_axis]
        mssim = np.empty(nch, dtype=float_type)

        if gradient:
            G = np.empty(im1.shape, dtype=float_type)
        if full:
            S = np.empty(im1.shape, dtype=float_type)
        channel_axis = channel_axis % im1.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)
        for ch in range(nch):
            ch_result = structural_similarity(im1[_at(ch)], im2[_at(ch)], **args)
            if gradient and full:
                mssim[ch], G[_at(ch)], S[_at(ch)] = ch_result
            elif gradient:
                mssim[ch], G[_at(ch)] = ch_result
            elif full:
                mssim[ch], S[_at(ch)] = ch_result
            else:
                mssim[ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return mssim, G, S
        elif gradient:
            return mssim, G
        elif full:
            return mssim, S
        else:
            return mssim
        
    # end of channel-looping case

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if gaussian_weights:
        # Set to give an 11-tap filter with the default sigma of 1.5 to match
        # Wang et. al. 2004.
        truncate = 3.5

    if win_size is None:
        if gaussian_weights:
            # set win_size used by crop to match the filter size
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
        else:
            win_size = 7  # backwards compatibility

    # SSIM 的窗口大小要小於影像尺寸
    if np.any((np.asarray(im1.shape) - win_size) < 0):
        raise ValueError(
            'win_size exceeds image extent. '
            'Either ensure that your images are '
            'at least 7x7; or pass win_size explicitly '
            'in the function call, with an odd value '
            'less than or equal to the smaller side of your '
            'images. If your images are multichannel '
            '(with color channels), set channel_axis to '
            'the axis number corresponding to the channels.'
        )
    # 窗口尺寸必須為奇數
    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')
    # 
    if data_range is None:
        if np.issubdtype(im1.dtype, np.floating) or np.issubdtype(
            im2.dtype, np.floating
        ):
            raise ValueError(
                'Since image dtype is floating point, you must specify '
                'the data_range parameter. Please read the documentation '
                'carefully (including the note). It is recommended that '
                'you always specify the data_range anyway.'
            )
        if im1.dtype != im2.dtype:
            warn(
                "Inputs have mismatched dtypes. Setting data_range based on im1.dtype.",
                stacklevel=2,
            )
        dmin, dmax = dtype_range[im1.dtype.type]
        data_range = dmax - dmin
        if np.issubdtype(im1.dtype, np.integer) and (im1.dtype != np.uint8):
            warn(
                "Setting data_range based on im1.dtype. "
                + f"data_range = {data_range:.0f}. "
                + "Please specify data_range explicitly to avoid mistakes.",
                stacklevel=2,
            )

    ndim = im1.ndim

    if gaussian_weights:
        filter_func = gaussian
        filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    im1 = im1.astype(float_type, copy=False)
    im2 = im2.astype(float_type, copy=False)

    NP = win_size**ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux**2 + uy**2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim. Use float64 for accuracy.
    mssim = crop(S, pad).mean(dtype=np.float64)

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(A1 / D, **filter_args) * im1
        grad += filter_func(-S / B2, **filter_args) * im2
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D, **filter_args)
        grad *= 2 / im1.size

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim