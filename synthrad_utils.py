import numpy as np
from PIL import Image
import nibabel as nib 
from glob import glob
from dicom_utils import (
    thresh_mask,
    single_wavelet_2d,
    to_uint8_255,
    find_body,
    normalize,
)
from training_utils import (
    illumination_correction,
    figure_combine
)
from typing import (
    Any,
    List,
    Union,
    Optional,
    Dict,
    Type,
    Tuple,
)
from cv2 import resize
from random import randint





### 這裡要改 ###
BASE_SYNTHRAD = 'SynthRAD2023/pelvis/'
###############




FOLDER_CONTENT = [ 'cbct.nii.gz', 'ct.nii.gz', 'mask.nii.gz' ]
def all_nii( base_path : str = BASE_SYNTHRAD ) -> List[ str ]:
    """
    all_nii:
    回傳所有 SynthRad2023 competition task2 中 nii 檔案位置。
    注意：目前僅使用 pelvis 資料集，brain 資料及有 anonymous process 造成的遮蓋問題

    Args:
    ----------
    base_path: str, default : 'SynthRAD2023/pelvis/'
    pelvis 資料夾基底位置。
    注意：`all_nii` 透過 `glob.glob` 完成任務，為了格式完整請在結尾加上 `/`
    
    Return:
    ----------
    List[ str ]
    所有 nii 子資料集位置，資料夾內存有 ct, cbct, mask 等三個 nii 檔，後續交由 `read_nii` 進行作業
    """
    nii_folders = glob( base_path + '*' )
    nii_folders.sort()

    return nii_folders

def read_nii( 
    nii_folders : List[ str ] = None, 
    index : int = 1,
    folder_name : Optional[ Union[ str, None ] ] = None 
) -> Union[ Dict[ str, np.ndarray ], None ]:
    """
    read_nii:
    依照給予的子資料夾位置讀取 `ct`, `cbct` 與 `mask` 等三個 nii 檔，以 np.ndarray 格式儲存

    Args:
    ----------
    nii_folder: List[ str ]
    所有子資料集位置
    ----------
    index: int, default : 1
    要讀取的檔案編號
    ----------
    filename: Optional[ str ]
    指定的子資料夾位置

    Return:
    ----------
    Dict[ str, np.ndarray ] or None
    key 順序：
    'cbct.nii.gz', 'ct.nii.gz', 'mask.nii.gz'
    """
    
    nii_contents : Dict[ str, np.ndarray ] = {}
    for name in FOLDER_CONTENT:
        if folder_name is None:
            filename = nii_folders[ index ] + '/' + name
        else :
            filename = folder_name + '/' + name
        try:
            content : nib.nifti1.Nifti1Image = nib.load( filename )
        except:
            print("su, read_nii: no file named: {}".format( filename ))
            return None
        nii_contents[ name ] = content.get_fdata() # returning np.ndarray

    return nii_contents

def to_pil_rotate(
    image : np.ndarray,
    deg_rotate : int = -90,
) -> Image.Image:
    """
    to_pil_rotate:
    將影像轉為 PIL.Image 並逆時針旋轉 90 度。
    SynthRAD2023 的 pelvis 橫切面需要進行旋轉。

    Args:
    ----------
    image: np.ndarray
    待轉換的影像資料
    ----------
    deg_rotate: int, default: -90
    旋轉角度，負號為逆時針方向。

    Return:
    ----------
    Image.Image
    轉換後影像

    """
    image = to_uint8_255( image )
    image : Type[ Image.Image ] = Image.fromarray( image ).convert( 'L' ).resize( ( 512, 512 ) )
    image = image.rotate( deg_rotate )
    return image

def store_image(
    image : Union[ np.ndarray, Type[ Image.Image ] ],
    filename : str,
    deg_rotate : Optional[ Union[ int, None ] ] = -90,
) -> None:
    """
    store_image:
    儲存影像(接受 `np.ndarray` 和 `Image.Image`)

    Args:
    ----------
    image: Union[ np.ndarray, Type[ Image.Image ] ]
    待儲存的影像資料
    ----------
    filename: str
    檔名
    ----------
    deg_rotate: Optional[ Union[ int, None ] ], default: -90
    旋轉角度。
    注意：SynthRAD2023 task2 pelvis dataset 中，橫切面影像需要逆時針旋轉 90 度

    Return:
    ----------
    None

    """
    if isinstance( image, np.ndarray ):
        image = image - image.min()
        image = image / image.max()
        image = np.uint8( image * 255 )
        image = Image.fromarray( image, 'L' )
    
    if deg_rotate is not None:
        image = image.rotate( deg_rotate )

    image.save( filename )
    return 

def ct_preprocessing( 
    image : np.ndarray,
    switch : bool = False,
    air_thresh : Optional[ int ] = -300,
    bone_thresh : Optional[ int ] = 200,
    tissue_thresh : Optional[ int ] = 0,
    axis : Optional[ int ] = 2, 
    ret_type : Optional[ str ] = 'image',
    pos_slice : Optional[ int ] = None,
    flag_hist : Optional[ bool ] = False,
    ) -> Dict[ str, Any ]:
    """
    ct_preprocessing:
    前處理，將 3 維的 np.ndarray 切出一張 2 維的 image

    Args:
    ----------
    image: np.ndarray
    待處理影像，3D np.ndarray
    ----------
    switch: bool, default: False
    是否需要套上 mask
    ----------
    thresh_val: int, default: 50
    mask 裁切閾值
    ----------
    axis: int, default: 2
    選定一個軸進行裁切，這裡用 2 可以得到(水平)橫切面
    ----------
    ret_type: Optional[ str ], default: `image`
    決定回傳樣式，目前支援：
    `image`, `mask`

    Return:
    ----------
    Dict[ str, Any ]
    `image` : np.ndarray, 經過裁切與前處理的資料
    `range` : List[ float ], 值域，maxima & minima
    `pos_slice` : int, 切片編號
    """

    if ret_type not in [ 'image', 'mask' ]:
        ret_type = 'image'

    if pos_slice is not None:
        # 要改這裡
        pos = pos_slice % ( image.shape[ axis % 3 ] )
        # pos = pos_slice % ( image.shape[ axis % 3 ] - 20 )
        # pos += 10
    else:
        pos = int( image.shape[ axis % 3 ] / 2 )

    if axis % 3 == 0:
        image = image[ pos, : , : ]
    elif axis % 3 == 1:
        image = image[ : , pos, : ]
    elif axis % 3 == 2:
        image = image[ : , : , pos ]

    # 儲存最初的數值範圍
    minima = image.min()
    maxima = image.max()
    air_mask = thresh_mask( image = image, thresh_val = air_thresh, mode = 'bigger' )
    store_image( air_mask, 'exp0521-ctps-air.png' )
    bone_mask = thresh_mask( image = image, thresh_val = bone_thresh, mode = 'bigger' )
    if ( flag_hist is True ) :
        img = illumination_correction( image = image, mask = air_mask, output_type = 'np', flag_hist = False )
        # tissue_mask = thresh_mask( image = img, thresh_val = tissue_thresh, mode = 'bigger' )
        tissue_mask = img
    else :
        tissue_mask = thresh_mask( image = image, thresh_val = tissue_thresh, mode = 'bigger' )
    image = image - image.min()
    image = image / image.max()
    image = image * 255

    image = illumination_correction( image, mask = air_mask, output_type = 'np' )
    

    if ( len( image.shape ) == 3 ):
        print( 'su-246, image shape is 3-ch' )
        
    if switch:
        image = image * bone_mask

    image = np.uint8( image )

    result = {
        "range" : [ maxima, minima ],
        "pos_slice" : pos,
        "air" : air_mask,
        "bone" : bone_mask,
        "tissue" : tissue_mask,
    }

    if ret_type == 'image':
        result[ "image" ] = image
    elif ret_type == 'mask':
        result[ "image" ] = bone_mask
    
    return result

def check_size(
    index : int = 1,
    axis : Optional[ int ] = 2,
    content_idx : Optional[ int ] = 1,
    ds_dir : Optional[ str ] = None,
) -> int:
    # 讀檔
    folders : List[ str ] = all_nii()
    
    content : Dict[ str, np.ndarray ] = read_nii( nii_folders = folders, index = index, folder_name = ds_dir )
    if content is None:
        return
    
    image : np.ndarray = content[ FOLDER_CONTENT[ content_idx ] ]

    if axis % 3 == 0:
        size = image.shape[ 0 ]
    elif axis % 3 == 1:
        size = image.shape[ 1 ]
    elif axis % 3 == 2:
        size = image.shape[ 2 ]

    return size

def synthrad_pipeline(
    index : int = 1,
    axis : Optional[ int ] = 2,
    content_idx : Optional[ int ] = 1,
    ds_dir : Optional[ str ] = None,
    pos_slice : Optional[ int ] = 30,
    rand_rotate : Optional[ bool ] = False,
    fix_rotate : Optional[ int ] = 0,
) -> Dict[ str, Union[ List[ Image.Image ], np.ndarray, int, List[ float ] ] ]:
    """
    synthrad_pipeline:
    讀取檔案與前處理

    Args:
    ----------
    index: int, default: 1
    nii 編號
    ----------
    axis: Optional[ int ], default: 2
    切面，預設為 2 ，是橫切面
    ----------
    content_idx: Optional[ int ], default: 1
    要讀取哪一種資料，有以下幾種模式：
    0: CBCT
    1: CT
    2: Mask
    ----------
    ds_dir: Optional[ str ], default: None
    指定的 nii 子資料集位置，須符合 `content_idx` 規定的格式
    ----------
    rand_rotate: Optional[ bool ], default: False
    是否要隨機旋轉輸出的資料，為了 augmentation 用途


    Return:
    ----------
    Dict[ str, Any ] or None:
    `images` : List[ Image.Image ]
        回傳一組影像，包含：
        0: filtered_ct,
        1: air
        2: bone
        3: wavelet
        4: tissue_mask

    `image` : np.ndarray, ct_preprocessing 執行結果
    `pos_slice` : int, 切片編號
    `range` : List[ float ], maxima & minima

    若沒有資料(讀檔錯誤)，則回傳 None
    """
    # 讀檔
    folders : List[ str ] = all_nii()
    
    content : Dict[ str, np.ndarray ] = read_nii( nii_folders = folders, index = index, folder_name = ds_dir )
    if content is None:
        return

    processed_ct : Dict[ str, Any ] = ct_preprocessing(
        content[ FOLDER_CONTENT[ content_idx ] ],
        axis = axis,
        switch = False,
        pos_slice = pos_slice
    )

    filtered_ct = to_uint8_255( processed_ct[ "image" ] )
    
    
    air = to_uint8_255( thresh_mask( image = filtered_ct, thresh_val = 50 ) )
    body = np.uint8( normalize( find_body( img = air ) ) )
    bone = to_uint8_255( thresh_mask( image = filtered_ct, thresh_val = 150 ) )
    tissue_mask = to_uint8_255( thresh_mask( filtered_ct, thresh_val = 110, mode = 'bigger' ) )

    filtered_ct *= body
    air *= body
    bone *= body
    tissue_mask *= body

    wavelet, hh = single_wavelet_2d( image = filtered_ct )
    wavelet = to_uint8_255( wavelet )
    
    images = [ filtered_ct, air, bone, wavelet, tissue_mask ]

    if rand_rotate is True:
        rotate_angle = -90 + randint( -180, 180 )
    else:
        rotate_angle = -90 + fix_rotate

    for idx, img in enumerate( images ):
        img = Image.fromarray( img ).convert( 'RGB' ).resize( ( 512, 512 ) )
        img = img.rotate( rotate_angle )
        images[ idx ] = img
    
    processed_ct[ "images" ] = images
    return processed_ct

def mask_operation(
    image : np.ndarray,
    pos_slice : int,
    axis : Optional[ int ] = 2,
) -> np.ndarray:
    if pos_slice is not None:
        # 要改這裡
        pos = pos_slice % ( image.shape[ axis % 3 ] )
        # pos = pos_slice % ( image.shape[ axis % 3 ] - 20 )
        # pos += 10
    else:
        pos = int( image.shape[ axis % 3 ] / 2 )

    if axis % 3 == 0:
        image = image[ pos, : , : ]
    elif axis % 3 == 1:
        image = image[ : , pos, : ]
    elif axis % 3 == 2:
        image = image[ : , : , pos ]

    mask = thresh_mask( image = image, thresh_val = 0, mode = 'bigger' ) # 0~1, np.uint8
    return mask

def pair_pipeline(
    index : int = 1,
    axis : Optional[ int ] = 2,
    ds_dir : Optional[ str ] = None,
    pos_slice : Optional[ int ] = 30,
    rand_rotate : Optional[ bool ] = False,
    fix_rotate : Optional[ int ] = 0,
    ct_params : Optional[ Dict[ str, Any ] ] = None,
    cbct_params : Optional[ Dict[ str, Any ] ] = None,
) -> Union [ 
        Dict[ str, Dict[ str, Union[ 
                    List[ Image.Image ], 
                    np.ndarray, 
                    int, 
                    List[ float ] ] ] ],
        None ]:
    """
    pair_pipeline:
    讀取檔案與前處理，會回傳一對資料

    Args:
    ----------
    index: int, default: 1
    nii 編號
    ----------
    axis: Optional[ int ], default: 2
    切面，預設為 2 ，是橫切面
    ----------
    ds_dir: Optional[ str ], default: None
    指定的 nii 子資料集位置，須符合 `content_idx` 規定的格式
    ----------
    rand_rotate: Optional[ bool ], default: False
    是否要隨機旋轉輸出的資料，為了 augmentation 用途


    Return:
    ----------
    Dict[ str, Dict[ str, Union[ List[ Image.Image ], np.ndarray, int, List[ float ] ] ] ] or None:
    第一層級 Dict 分為 `ct`, `cbct` 與 `mask`;
    第二層級 Dict:
    `images` : List[ Image.Image ]
        回傳一組影像，包含：
        0: filtered_ct,
        1: air
        2: bone
        3: wavelet
        4: tissue_mask

    `image` : np.ndarray, ct_preprocessing 執行結果
    `pos_slice` : int, 切片編號
    `range` : List[ float ], maxima & minima

    若沒有資料(讀檔錯誤)，則回傳 None
    """
    # 讀檔
    folders : List[ str ] = all_nii()
    
    content : Dict[ str, np.ndarray ] = read_nii( nii_folders = folders, index = index, folder_name = ds_dir )
    if content is None:
        return None

    def image_processing(
        content_idx : Optional[ int ] = 1, # 0: cbct, 1: ct
        bone_thresh : Optional[ int ] = 200,
        air_thresh : Optional[ int ] = -300,
        tissue_thresh : Optional[ int ] = 0,
        bone_enhance : Optional[ float ] = 0.1, # 有加會表現得比較好
        flag_hist : Optional[ bool ] = False,
    ) -> Dict[ str, Any ]:
        processed_ct : Dict[ str, Any ] = ct_preprocessing(
            content[ FOLDER_CONTENT[ content_idx ] ],
            axis = axis,
            switch = False,
            pos_slice = pos_slice,
            air_thresh = air_thresh,
            bone_thresh = bone_thresh,
            tissue_thresh = tissue_thresh,
        )
        """
        processed_ct:
        `image` : np.ndarray, CT slice
        `pos_slice` : int, slice number of given image
        `range` : List[ float ], [ maxima, minima ]
        """
        filtered_ct = to_uint8_255( processed_ct[ "image" ] )
        if ( len( filtered_ct.shape ) == 3 ):
            print( 'su-492, index: {} has 3-ch filtered_ct !!'.format( index ) )
        
        if ( np.max( filtered_ct ) == 0 ):
            print( 'Invalid frame in patient {}, content {}, slice {}'.format( index, content_idx, pos_slice ) )
            return None
    
        # 取得 mask
        
        air = to_uint8_255( processed_ct[ "air" ] )
        store_image( air, 'exp0521-air-mask.png' )
        if content_idx == 0:
            store_image( air, filename = 'synth0505-cond-air.png' )
            # print('value range: {} ~ {}'.format( processed_ct[ "range" ][ 0 ], processed_ct[ "range" ][ 1 ] ) )
            pass
        body = np.uint8( normalize( find_body( img = air ) ) )
        store_image( body, 'exp0521-body-mask.png' )
        bone = to_uint8_255( processed_ct[ "bone" ] )
        tissue_mask = to_uint8_255( processed_ct[ "tissue" ] )
        # 僅留下身體
        if ( content_idx == 0 ):
            store_image( filtered_ct, 'exp0521-filtered_ct.png')
        filtered_ct *= body 
        air *= body
        bone *= body
        tissue_mask *= body
        
        # bone enhancement
        filtered_ct = np.float32( filtered_ct )
        bone = np.float32( bone )
        
        filtered_ct = filtered_ct + bone_enhance * bone
        filtered_ct = to_uint8_255( filtered_ct )
        bone = to_uint8_255( bone )

        wavelet, hh = single_wavelet_2d( image = filtered_ct )
        wavelet = to_uint8_255( wavelet )
        
        images = [ filtered_ct, air, bone, wavelet, tissue_mask ]

        if rand_rotate is True:
            rotate_angle = -90 + randint( -180, 180 )
        else:
            rotate_angle = -90 + fix_rotate

        for idx, img in enumerate( images ):
            img = Image.fromarray( img ).convert( 'RGB' ).resize( ( 512, 512 ) )
            img = img.rotate( rotate_angle )
            images[ idx ] = img
        
        return { "images" : images, "processed_dict" : processed_ct }


    if cbct_params is not None:
        cbct_dict : Dict[ str, Any ] = image_processing( 
            content_idx = 0, 
            bone_thresh = cbct_params[ 'bone' ],
            air_thresh =  cbct_params[ 'air' ],
            tissue_thresh = cbct_params[ 'tissue' ],
            flag_hist = True,
            bone_enhance = 0 )

    else:
        cbct_dict : Dict[ str, Any ] = image_processing( 
            content_idx = 0, 
            bone_thresh = 1000,
            air_thresh = -550,
            tissue_thresh = 750,
            flag_hist = True, )
        
    ct_dict : Dict[ str, Union[ List[ Image.Image ], Dict[ str, Any ] ] ] = image_processing( 
        content_idx = 1, 
        bone_thresh = 200,
        air_thresh = -300,
        tissue_thresh = 0, )

    mask = mask_operation( image = content[ FOLDER_CONTENT[ 2 ] ], pos_slice = pos_slice )
    mask = resize( mask.astype( float ), ( 512, 512 ) )
    mask = np.uint8( mask )
    
    return { "ct" : ct_dict, "cbct" : cbct_dict, "mask" : mask }

def test_0507():
    # 測試 Task2 Pelvis set 中 Site B 的資料究竟要用甚麼 threshold value
    offset = 60
    portion = 200/60
    pos = 20
    for i in range( 60 ):
        patient = offset + i
        
        cbct_par = {
            "air" : -550,
            "bone" : -195,
            "tissue" : -300,
        }
        results = pair_pipeline( index = patient, axis = 2, pos_slice = pos, rand_rotate = False, cbct_params = cbct_par )
        
        ct : Dict[ str, Any ] = results[ "ct" ]
        """
        figure_combine(
            images = {
                "CT" : ct[ "images" ][ 0 ],
                "Air" : ct[ "images" ][ 1 ],
                "Bone" : ct[ "images" ][ 2 ],
                "Tissue" : ct[ "images" ][ 4 ],
            },
            figname = "synth0426-{}-{}-ct.png".format( patient, pos ),
            plotname = "New preprocessing on {}-{}".format( patient, pos ),
        )
        """
        cbct : Dict[ str, Any ] = results[ "cbct" ]
        figure_combine(
            images = {
                "CT" : cbct[ "images" ][ 0 ],
                "Air{}".format( cbct_par["air"] ) : cbct[ "images" ][ 1 ],
                "Bone{}".format( cbct_par["bone"] ) : cbct[ "images" ][ 2 ],
                "Bone(CT)" : ct[ "images" ][ 2 ],
            },
            figname = "synth0426-{}-{}-cbct.png".format( patient, pos ),
            plotname = "New preprocessing on {}-{}".format( patient, pos ),
        )

if __name__ == '__main__' :

    print( 'Current base path: {}'.format( BASE_SYNTHRAD ) )
    print( 'Modify it if neccessary.')

    """
    synthrad_pipeline(
        index = 1, # 子資料夾編號，注意整個 task2 所有的資料一起拉進來編號了
        axis = 2, # 切在哪一軸，建議開 2 ，是橫切面
        content_idx = 1, # 1 是 CT、0 是 CBCT
        ds_dir = None, # 預設 None ，也可以指定子資料夾，例如：".../SynthRAD2023/pelvis/2PA032"，此時會忽略 `index`
        pos_slice = 30, # 在 cubic 中取第幾片 slice
        rand_rotate = False, # 是否隨機旋轉
        fix_rotate = 0 # 固定旋轉角度
    )
    """
    """
    POS = 20
    INDEX = 115
    cbct_par = {
        "air" : -550,
        "bone" : -200,
        "tissue" : -300,
    }
    results = pair_pipeline( index = INDEX, axis = 2, pos_slice = POS, rand_rotate = False, cbct_params = cbct_par )
    ct : Dict[ str, Any ] = results[ "ct" ]
    figure_combine(
        images = {
            "CT" : ct[ "images" ][ 0 ],
            "Air" : ct[ "images" ][ 1 ],
            "Bone" : ct[ "images" ][ 2 ],
            "Tissue" : ct[ "images" ][ 4 ],
        },
        figname = "synth0426-{}-{}-ct.png".format( INDEX, POS ),
        plotname = "New preprocessing on {}-{}".format( INDEX, POS ),
    )
    cbct : Dict[ str, Any ] = results[ "cbct" ]
    figure_combine(
        images = {
            "CT" : cbct[ "images" ][ 0 ],
            "Air" : cbct[ "images" ][ 1 ],
            "Bone" : cbct[ "images" ][ 2 ],
            "Tissue" : cbct[ "images" ][ 4 ],
        },
        figname = "synth0426-{}-{}-cbct.png".format( INDEX, POS ),
        plotname = "New preprocessing on {}-{}".format( INDEX, POS ),
    )
    """
    test_0507()
    pass