# 2023-07-31

import numpy as np
import cv2
import torch
import os
from math import ceil
import scipy.signal as signal
from PIL import Image 
from pydicom import dcmread 
from pydicom.dicomdir import DicomDir
from typing import Tuple, List, Union, Callable, Any, Dict, Optional
from glob import glob
from matplotlib import pyplot as plt


import pywt

#### 將輸入的 np.ndarray 正規化到 0 ~ 1 之間 ####
def normalize( array: np.ndarray ) -> np.ndarray:
    if ( np.max( array ) == np.min( array ) ):
        # 數值都一樣
        return np.zeros_like( array, dtype = array.dtype )
    array = array - np.min( array )
    array = array / np.max( array )
    return array
#### 將輸入的 np.ndarray 以 np.uint8 格式正規化到 0-255 之間 ####
def to_uint8_255( arr: np.ndarray ) -> np.ndarray:
    arr = normalize( arr )
    arr = np.uint8( arr * 255 )
    return arr

def test_save_img( 
    array : np.ndarray,
    name: str
) -> None:

    img = Image.fromarray( to_uint8_255( array ), 'L' ).convert( 'RGB' )
    img.save( name )
    return 

#### DICOM / HU image ####
# 將影像轉換為 hu 
# 圖片基本構型與對比幾乎沒變
# 只是改變圖片值域，就是個線性平移
def transform_to_hu(medical_image: DicomDir, image: np.ndarray ) -> np.ndarray:
    # :param medical_image: dicom content
    # :param image: dicom.pixel_array
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

#### 輸入 CT 影像與闕值，輸出 0 or 1 構成的 mask ####
# 輸出值域為 0-1 ( 畢竟只有 0, 1 可以填 )
def thresh_mask( image: np.ndarray, thresh_val: int = -900, mode: str = 'bigger' ) -> np.ndarray:
    """
    thresh_mask:
    輸出由 0, 1 組成的 mask

    Args:
    ---------
    image: np.ndarray
    待處理影像，mask 的生成將基於 `image` 的 pixel value
    ----------
    thresh_val: int
    閾值，注意這裡判定時不包含等於 thresh_val 的部分
    ---------
    mode: str
    比大還比小，目前支援 `smaller` and `bigger`

    Return:
    ---------
    np.ndarray
    輸出由 0, 1 構成的 mask
    """
    
    # 確認 threshold 模式
    if mode not in [ 'bigger', 'smaller' ]:
        print( 'thresh_mask: invalid mode: {}, default to \"bigger\"'.format( mode ) )
        mode = 'bigger'
    
    # 以固定 threshold value 進行 HU 值篩選
    if mode == 'smaller':
        ret = np.where( image < thresh_val, 1, 0 )
    elif mode == 'bigger':
        ret = np.where( image > thresh_val, 1, 0 )

    return ret

def hu_clip(
    array : np.ndarray,
    mask : np.ndarray,
    upper : Optional[ int ] = 500,
    lower : Optional[ int ] = -300,
) -> np.ndarray:
    
    upper_mask = thresh_mask( array, upper, 'smaller' )
    upper_comp = thresh_mask( array, upper, 'bigger' )
    lower_mask = thresh_mask( array, lower, 'bigger' )

    comb_mask = lower_mask*upper_mask*mask
    
    comb_inv = thresh_mask( comb_mask, thresh_val = 1, mode = 'smaller' )
    comp_inv = thresh_mask( upper_comp, thresh_val = 1, mode = 'smaller' )

    array = ( array * comb_mask ) + upper_comp*mask*upper + comb_inv*comp_inv*lower

    return array

def find_body( img: np.ndarray ) -> np.ndarray:
    """
    find_body:
    計算 CT 影像中的體腔位置

    Args:
    ----------
    img: np.ndarray
    輸入 CT 影像

    Return:
    ----------
    np.ndarray
    回傳 mask ，為 0 or 255 的 mask
    """

    if ( np.max( img ) == 0 ):
        ret = np.zeros_like( img, dtype = np.uint8 )
        return ret

    # 圖片需要配合 cv2 的格式，改為 np.uint8 格式，值域為 0-255
    # 這裡使用的圖片為 no_air ，經過 np.where 出來，是 0-1 的 float
    # 故需轉換型態與值域
    img = img - np.min( img )
    img = img / np.max( img )
    img = np.uint8( img * 255 )
    # template 是 img 的三通道版本，用於 cv2.drawContours()
    # => 一個通道的圖是不能畫上顏色的
    template = np.zeros( img.shape, dtype = np.uint8 )
    template = cv2.cvtColor( template, cv2.COLOR_GRAY2BGR )

    # 找出每個輪廓，並篩選出真的是 body 的 contour 
    # => ( 一個或多個 )
    contours, hierarchy = cv2.findContours( img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    
    body_result = []
    for idx, contour in enumerate( contours ):
        # 若這個 contour 是依附在某個 parent contour 之下，則不進行判定
        # 因為 body 不會是某個 parent contour 之內的 child contour
        if hierarchy[ 0 ][ idx ][ 3 ].item() != -1:
            continue

        # 2023-08-22: 有些床鋪的面積大於 10000，新增位置判定
        # 輪廓位置太低者，一律移除
        y_coors: np.ndarray = contour[ : , 0, 1 ]
        y_central = sum( y_coors )/contour.shape[ 0 ]
        if y_central > 350:
            continue
        
        # 計算 contour 圍出的面積，若小於 threshold value 就不算
        # => 這裡設為 10000 
        area = cv2.contourArea( contour )
        if area > 10000:
            body_result.append( contour )

    # 繪製出 contour 的位置，製成 mask 
    cv2.drawContours( template, body_result, -1, ( 255, 255, 255 ), thickness = cv2.FILLED )
    # display( Image.fromarray( template ).convert( "RGB" ) )

    return template[ : , : , 0 ]

#### 輸入 CT 影像，輸出由 sobel operators 找出的影像邊緣 ####
# 輸出值域為 0-1
def gen_sobel( image: np.ndarray ) -> np.ndarray:
    # 以 sobel operator 凸顯輪廓
    filter_x = np.array( [ [ -1, 0, 1 ], [ -2, 0, 2 ], [ -1, 0, 1 ] ], dtype = np.float64 )
    filter_y = filter_x.transpose()

    sx = signal.convolve2d( image, filter_x, mode = "same", boundary = "symm", fillvalue = 0 )
    sy = signal.convolve2d( image, filter_y, mode = "same", boundary = "symm", fillvalue = 0 )
    sobel = np.hypot( sx, sy )
    sobel = normalize( sobel )
    return sobel

def histo_inspect( arr: np.ndarray, 
                flag_plot: bool = False, 
                bin_factor: int = 2, 
                r_start: int = 0, 
                r_end: int = -1 ) -> np.ndarray:
    """
    histo_inspect:
    統計 0-255 每種 pixel-value 的數量，由 bin_factor 可以影響 bin 的密集程度。
    作為 tissue condition pipeline 中的第一層線性量化，bin_factor 可以直接影響到量化的程度

    Args:
    ----------
    arr: np.ndarray
    待量化的輸入矩陣
    ----------
    flag_plot: bool
    是否要顯示出 histogram
    預設是不要，不希望造成常規前處理流水線有額外的麻煩
    ----------
    bin_factor: int
    用於決定量化區間的常數，數字愈大，bin 愈少，每個 bin 代表的區間愈寬
    ----------
    r_start: int
    顯示 histogram 時，指定圖的左側邊界，可以避免顯示出多餘的 bin ，聚焦在關注的範圍中
    ----------
    r_end: int
    作用同 r_start ，用於決定右側邊界

    Return:
    ----------
    hist: np.ndarray
    統計 arr 的 histogram 

    """
    # 以 histogram 統計輸入影像的數值表現
    n_bins = ( ( np.max( arr ) - np.min( arr ) ) // bin_factor ) + 1
    n_bins = np.int32( n_bins )
    hist, bin_edges = np.histogram( arr, n_bins )
    # bin_edges, len: ( hist ) + 1
    x_bar = range( np.int32( np.min( arr ) ), np.int32( np.max( arr ) ) )[ :: bin_factor ]
    skip_factor = 8
    
    if flag_plot is True:
        plt.figure()
        plt.plot( x_bar[ r_start: r_end ][ :: skip_factor ], hist[ r_start: r_end ][ :: skip_factor ] )
        plt.show()
        plt.close()
    
    return hist



def gen_tissue( 
    q_factor    : int, 
    c_factor    : int, 
    target_ct   : np.ndarray, 
    air_mask    : np.ndarray, 
    bone_mask   : Optional[ np.ndarray ] = None,
    top_k       : int = 5,
    flag_bone   : Optional[ bool ] = False ) -> np.ndarray:
    """
    gen_tissue:
    基於非線性量化( Non-Linear Quantization )製作的 tissue condition ，
    透過 q_factor 與 c_factor 調控量化的程度，
    最後以 top_k 的形式標示出不同顏色深淺的區域

    Args:
    ----------
    q_factor: int
    第一階段線性量化的區間大小，
    值域已知為 [ 0-255 ]，實際量化時的 bin 數量為 256 // q_factor，
    故可知 q_factor 愈大，第一階段量化的程度愈高，失真度愈低
    ----------
    c_factor: int
    第二階段非線性量化的區間大小，
    由於組織部位的顏色數值集中出現在 100-150 ，故在這一區間內的 bin 常常佔據 top_k 排行，
    為了讓其他區域的組織可以排進 top_k 被凸顯出來，可以將這一區域內的 bin 數量進一步減少，
    除了可以讓其他 bin 進入 top_k ，也能絕對保證這一區間一定會有 bin 進入 top_k 排行
    ----------
    target_ct: np.ndarray
    待處理 CT 影像，需要先經過去除床鋪的操作，
    理論上應該是 pipeline 中的 filtered_ct
    ----------
    air_mask: np.ndarray
    與 target_ct 對應的空腔遮罩，用於在最後清出空洞
    ----------
    bone_mask: np.ndarray
    與 target_ct 對應的骨骼遮罩，用於凸顯骨骼位置
    ----------
    sobel: np.ndarray
    與 target_ct 對應的 sobel edges，用於 edge enhancement
    ----------
    top_k: int
    不同色塊種類的數量

    Return:
    ----------
    template: np.ndarray
    用於控制生成的 tissue condition 
    值域非 [ 0, 1 ], [ 0, 255 ]，需要 normalize 
    """

    # 先透過模糊盡可能抹除 artifact ，轉換至 0-255 並統計各 value 的出現位置
    target_ct = cv2.bilateralFilter( to_uint8_255( target_ct ), 7, sigmaSpace = 75, sigmaColor = 75 )
    target_ct = to_uint8_255( target_ct )
    if target_ct.max() < 255:
        print( 'dicom_utils, gen_tissue: invalid target_ct, returning bad result' )
        return target_ct
    histo = histo_inspect( arr = target_ct, bin_factor = q_factor, flag_plot = False )
    
    # 以非均勻量化( Non-uniform quantization )處理 histogram ，此舉可以降低峰值集中處的大量出現導致大片區域對比度降低
    # 舉例來說，CBCT 在經過以上前處理後，非 0 數值大多聚集在 100-150( 範圍是 0-255 )，
    # 而這些數值就是體腔的顏色，
    # 當直接使用這個 histogram 的前幾個峰值時，會因為這個範圍的數值大量出現導致分區效果不佳，
    # 體腔被過分地分割出多種顏色，像是 120, 122 分別佔據 top_k 中的兩個位置，導致有不同顏色，但肉眼看起來卻一樣，
    # 而其他特徵因為較小，就被埋沒在這些過分佔據 top_k 的重複元素之後

    # 使用非均勻量化，可以在特定的區域內進一步壓縮 bin 的數量，一方面保證它們一定會進 top_k ( 畢竟數量合併 )，
    # 另一方面能讓更多區域的數值進入 top_k
    bin_table = list( range( 0, 256, q_factor ) )
    alter_table = []
    alter_histo = []
    sum = 0
    cnt = 0
    for k in range( len( bin_table ) ):
        # 只進一步壓縮 100-150 這個區間
        if k*q_factor >= 100 and k*q_factor <=150:
            # 由 c_factor 控制壓縮的比例
            if cnt == c_factor - 1:
                alter_table.append( bin_table[ k ] )
                alter_histo.append( histo[ k ] + sum )
                # 刷新 buffer 
                sum = 0
                cnt = 0
            else:
                cnt += 1 # cnt 負責記住 buffer 了幾個 bin
                sum += histo[ k ] # sum 負責 buffer 這些 bin 的數據
                
        else:
            # 非壓縮範圍，value 與 amount 照填
            alter_table.append( bin_table[ k ] )
            alter_histo.append( histo[ k ] )
    
    # 由 q_factor 決定量化的間距
    # histo 統計了各個 bin 上有多少 pixel
    order = np.flip( np.argsort( alter_histo ), axis = 0 ) # order 為 histo 中元素的大小排序，由大到小，並且只取前 k 組
    template = np.zeros( shape = target_ct.shape ) # 產圖的基底
    
    access_pixelval : Callable[ [ int ], int ] = lambda idx : alter_table[ order[ idx ] ] # 透過 idx 找出對應的 pixel-value
    # 先透過 order 取得 alter_table 中的 index 
    # 再透過這個 index 取得 alter_table 中的元素( 特定 pixel-value 的數值 )
    
    # 將數量最多的幾個數值出現區域一層一層疊起來
    # 形成類似 semantic mask 的輸出
    for idx in range( top_k ):
        bucket = access_pixelval( idx ) # 循序讀取 histo 數據
        local_mask = np.where( target_ct > bucket, 1, 0 )
        template += local_mask
    # 最後加上 bone 
    bone_mask = normalize( bone_mask )
    bone_mask = np.where( bone_mask == 0, 0, 1 )
    # 去除空洞
    if flag_bone is True:
        template += bone_mask
    template *= air_mask

    return template

def sobel_alter( ds: DicomDir ) -> np.ndarray:
    hu = transform_to_hu( ds, ds.pixel_array )
    inv_bone_mask = thresh_mask( hu, thresh_val = 200, mode = 'smaller' )
    no_air = thresh_mask( hu, thresh_val = -300, mode = 'bigger' )
    mask = normalize( find_body( no_air ) )
    filtered_ct = to_uint8_255( ds.pixel_array ) * mask

    sobel = gen_sobel( filtered_ct )
    air_mask = no_air*mask
    sobel = sobel * inv_bone_mask * air_mask
    sobel = thresh_mask( image = sobel, thresh_val = 0.05, mode = 'bigger' )
    return sobel

def tissue_enhance( tissue: Image.Image, sobel: Image.Image ) -> Image.Image:
    """
    tissue_enhance:
    輸入 pipeline 生成的 tissue 與 sobel 後，以固定的權重輸出加強了 sobel edges 的 tissue condition

    Args:
    ----------
    tissue: Image.Image
    tissue condition
    ---------
    sobel: Image.Image
    sobel edges detection condition

    Return:
    ---------
    tissue: Image.Image
    強調了邊緣的 tissue condition
    """
    tissue = to_uint8_255( np.array( tissue.convert( 'L' ).resize( ( 512, 512 ) ) ) )
    sobel = to_uint8_255( np.array( sobel.convert( 'L' ).resize( ( 512, 512 ) ) ) )

    sobel_thresh = np.where( sobel > 50, 1, 0 )
    tissue = tissue + sobel_thresh*5
    tissue = Image.fromarray( to_uint8_255( tissue ) ).convert( 'RGB' ).resize( ( 512, 512 ) )
    return tissue

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
    coeffs = pywt.dwt2( image, wavelet )
    LL, ( LH, HL, HH ) = coeffs

    return LL, HH

def wavelet_recon(
    generated_signal : Union[ np.ndarray, Image.Image ],
    baseline_signal : Union[ np.ndarray, Image.Image ],
    level : Optional[ int ] = 2,
    wavelet : Optional[ str ] = 'bior2.6',
    ) -> Image.Image:

    stack_LH = []
    stack_HL = []
    stack_HH = []

    if level < 1:
        level = 1
    if not isinstance( level, int ):
        level = ceil( level )
    for cnt in range( level ):
        coeffs = pywt.dwt2( generated_signal, wavelet )
        ll, ( lh, hl, hh ) = coeffs

        stack_LH.append( lh.copy() )
        stack_HL.append( hl.copy() )
        stack_HH.append( hh.copy() )

        generated_signal = ll

        coeffs = pywt.dwt2( baseline_signal, wavelet )
        ll, ( lh, hl, hh ) = coeffs
        baseline_signal = ll

    
    for cnt in range( level ):
        lh : np.ndarray = stack_LH.pop()
        hl : np.ndarray = stack_HL.pop()
        hh : np.ndarray = stack_HH.pop()
        
        if baseline_signal.shape != hh.shape:
            baseline_signal = cv2.resize( baseline_signal, hh.shape )
        baseline_signal = pywt.idwt2( ( baseline_signal, ( lh, hl, hh ) ), wavelet )
        
    baseline_signal = to_uint8_255( baseline_signal )
    return Image.fromarray( baseline_signal ).convert( "RGB" ).resize( ( 512, 512 ) )

def cond_pipeline( ds: DicomDir ) -> Union[ List[ np.ndarray ], None ]:
    hu = transform_to_hu( ds, ds.pixel_array )

    test_save_img( ds.pixel_array, "test_image.png" )

    if np.max( hu ) == np.min( hu ):
        # 若讀取的資料完全都是空的
        return None
    # 先由 Hounsfield Unit 切出 mask
    # -300 切出 air ( lucent area )
    # 200 切出 bone
    # 0 切出 tissue ( +100 是 soft tissue 的 upper bound ，但取 0 可以移除 bone 與部分 tissue 間的色差 )
    air_mask = thresh_mask( hu, thresh_val = -300, mode = 'bigger' )
    bone_mask = thresh_mask( hu, thresh_val = 200, mode = 'bigger' )
    tissue_mask = thresh_mask( hu, thresh_val = 0, mode = 'bigger' )
    # 由 body mask 切掉 bed
    body_mask = normalize( find_body( air_mask ) )
    filtered_ct = to_uint8_255( hu_clip( hu, body_mask ) )
    # filtered_ct = to_uint8_255( ds.pixel_array )*body_mask
    # 將 cond 轉到 np.uint8

    cond_air = to_uint8_255( air_mask * body_mask )
    cond_sobel = to_uint8_255( sobel_alter( ds ) )
    cond_bone = to_uint8_255( bone_mask * body_mask )
    cond_tissue = to_uint8_255( tissue_mask * body_mask )
    cond_body = to_uint8_255( body_mask )

    # 將黑色殘影透過 dilation 移除
    dilate_kernel = np.ones( ( 3, 3 ), np.uint8 )
    cond_tissue = cv2.dilate( cond_tissue, dilate_kernel, iterations = 1 )
    cond_tissue = to_uint8_255( cond_tissue )
    
    # 取出小波
    cond_wavelet, hh = single_wavelet_2d( image = filtered_ct )
    if len( cond_wavelet.shape ) >= 3:
        raise ValueError( 'dicom_utils, cond_pipeline: invalid wavelet condition' )
    cond_wavelet, hh = single_wavelet_2d( image = cond_wavelet )
    cond_wavelet = to_uint8_255( cond_wavelet )
    
    return [ filtered_ct, cond_air, cond_bone, cond_sobel, cond_tissue, cond_body, cond_wavelet ] # data type: np.uint8

def soft_tissue(
    ds_dir :str,
) -> List[ Image.Image ]:
    """
    soft_tissue:
    依照 HU > 0 產出軟組織條件

    Args:
    ----------
    ds_dir : str
    dicom 檔位置

    Return:
    ----------
    List[ Image.Image ]
    0: ct,
    1: tissue,
    2: tissue_dilate,
    3: cond_air ( 用於亮度校正 )
    """
    ds = dcmread( ds_dir )
    hu = transform_to_hu( ds, ds.pixel_array )
    if np.max( hu ) == np.min( hu ):
        # 若讀取的資料完全都是空的
        return None
    
    air_mask = thresh_mask( hu, thresh_val = -300, mode = 'bigger' )
    tissue_mask = thresh_mask( hu, thresh_val = 0, mode = 'bigger' )
    body_mask = normalize( find_body( air_mask ) )
    
    filtered_ct = to_uint8_255( ds.pixel_array )*body_mask
    cond_tissue = to_uint8_255( tissue_mask * body_mask )
    cond_air = to_uint8_255( air_mask * body_mask )

    tissue_mask = np.uint8( tissue_mask )

    dilate_kernel = np.ones( ( 3, 3 ), np.uint8 )
    tissue_dilate = cv2.dilate( cond_tissue, dilate_kernel, iterations = 1 )
    tissue_dilate = to_uint8_255( tissue_dilate )

    filtered_ct = Image.fromarray( filtered_ct ).convert( "RGB" ).resize( ( 512, 512 ) )
    cond_tissue = Image.fromarray( cond_tissue ).convert( "RGB" ).resize( ( 512, 512 ) )
    tissue_dilate = Image.fromarray( tissue_dilate ).convert( "RGB" ).resize( ( 512, 512 ) )
    
    return [ filtered_ct, cond_tissue, tissue_dilate, cond_air ]

def dicom_pipeline( 
    ds_dir: str, 
    flag_pil: bool = True,
    pos_slice: Optional[ int ] = 0,
    ) -> Union[ List[ Image.Image ], List[ np.ndarray ], None ]:
    """
    dicom_pipeline:
    將指定 dicom 檔中的 CT 讀出來，並透過前處理生成 input 與 conditions
    回傳前已正規化到 0-255 且在 RGB 空間中

    Args:
    ----------
    ds_dir: str
    指定 dicom 檔的位址，將透過 dcmread 開檔
    ----------
    flag_pil: bool
    決定要回傳的 image 是 PIL.Image.Image 還是 np.ndarray

    Return:
    ----------
    result: Union[ List[ Image.Image ], List[ np.ndarray ], None ]
    直接透過 cond_pipeline() 回傳一個 List
    2023-08-09 版本回傳順序如下：
    - 0 filtered_ct
    - 1 air_mask
    - 2 bone_mask
    - 3 sobel
    - 4 tissue
    - 5 body_mask
    - 6 wavelet

    也可能回傳 None ，表示畫面全黑
    """
    ds = dcmread( ds_dir )
    if flag_pil is True:
        images = cond_pipeline( ds )
        if images is not None:
            for idx, img in enumerate( images ):
                images[ idx ] = Image.fromarray( img ).convert( "RGB" ).resize( ( 512, 512 ) )

        return images
    else:
        return cond_pipeline( ds )

if __name__ == '__main__':
    # demo

    # 讀檔
    # ds_paths 需要是一個 List[ str ] ，每個元素都是真實存在的 dicom 檔位置
    ds_paths = glob( 'dataset/CHEST/*/CT/CT*.dcm' )
    # ds_paths = glob( 'dataset/CHEST/*/CBCT/CT*.dcm' )
    print( 'There are {} dicoms available.'.format( len( ds_paths ) ) )
    # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # to avoid OMP: Error #15
    
    # rets 是 List[ PIL.Image.Image ] 
    rets = dicom_pipeline( ds_dir = "dataset/CHEST/CHEST_2021_002/CT/CT.1.3.12.2.1107.5.1.4.29309.30000021061802142667100000863.dcm" )
    # "dataset/CHEST/CHEST_2021_001/CT/CT.1.3.12.2.1107.5.1.4.29309.30000021052600202517100000005.dcm"

    
    rets[ 0 ].save( "dicom_utils_new_hu-1.png" )
    rets[ 6 ].save( "dicom_utils_new_wavelet-1.png" )
    