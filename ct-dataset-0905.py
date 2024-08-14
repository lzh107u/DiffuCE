import datasets
from datasets.download.download_manager import DownloadManager
from PIL import Image
import numpy as np
from glob import glob
from typing import Tuple, Callable, Union, List, Dict, Any, Optional
import cv2
import pickle
from dicom_utils import to_uint8_255, gen_sobel, gen_tissue, thresh_mask, single_wavelet_2d, soft_tissue, dicom_pipeline
from DiffuCE import chest_set_parsing
from training_utils import illumination_correction

PICKLE_FOLDER = '0915_tuning/pickles'

_CITATION = """\
    @InProceedings{huggingface: dataset,
    title = {A dataset for ControlNet training},
    author = {lzh107u},
    year = {2023},
    }
    """

_DESCRIPTION = """\
    This dataset is to train a ControlNet with sobel edges from CT images.
    """
# not yet prepared
_HOMEPAGE = ""

def load_image_1222(
    cond_mode : str = 'tissue',
    data_dir : str = None,
) -> Tuple[ Image.Image ]:
    
    images = dicom_pipeline( ds_dir = data_dir )
    ct = images[ 0 ]
    
    if cond_mode == 'tissue':
        cond_image = images[ 4 ]
    elif cond_mode == 'bone':
        cond_image = images[ 2 ]
    elif cond_mode == 'air':
        cond_image = images[ 1 ]
    elif cond_mode == 'wavelet':
        cond_image = images[ 6 ]
    elif cond_mode == 'sobel':
        cond_image = images[ 3 ]
    else:
        raise ValueError("Invalid `cond_mode`:{} when training controlnet".format( cond_mode ) )
    
    return ct, cond_image

def load_image( 
    idx         : int = 200,
    cond_mode   : str = 'bone',
    data_dir    : Optional[ Union[ str, None ] ] = None,
    ) -> Tuple[ Image.Image, Image.Image ]:
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
    cond_mode: str
    指定要回傳的條件影像
    目前有以下模式：
    - air
    - bone
    - tissue
    ----------
    data_dir: str

    Return:
    ----------
    Tuple[ Image.Image, Image.Image ]
    回傳 unet 與 controlnet 的輸入影像

    """
    post_proc : Callable[ [ np.ndarray ], Image.Image ] = lambda arr : Image.fromarray( arr ).convert( "RGB" ).resize( ( 512, 512 ) )
    
    if data_dir is not None:
        path = data_dir
    else:
        path = '{}/ct{}.pkl'.format( PICKLE_FOLDER, idx )

    with open( path, 'rb' ) as f:
        data_dict : Dict[ str, Any ] = pickle.load( f )
        # 以下為 data_dict 各元素名稱與其對應內容物
        # filtered_ct: Image.Image, RGB, 512x512
        # name: str
        # hu: np.ndarray
        # cbct: Image.Image, RGB, 512x512
        # body_mask: Image.Image, RGB, 512x512
        data_dict : Dict[ str, Union[ Image.Image, np.ndarray ] ] = data_dict[ 'origin' ]
        
        filtered_ct : Image.Image = data_dict[ 'ct' ]
        hu : np.ndarray = data_dict[ 'hu' ]
        body_mask : Image.Image = data_dict[ 'body_mask' ]
        cbct : Image.Image = data_dict[ 'cbct' ]
        
    if cond_mode not in [ 'air', 'bone', 'tissue', 'body', 'wavelet' ]:
        cond_mode = 'bone'
    
    # 由 hu 值獲得空氣與骨骼部位
    air_mask = thresh_mask( hu, thresh_val = -300, mode = 'bigger' )
    bone_mask = thresh_mask( hu, thresh_val = 200, mode = 'bigger' )
    tissue_mask = thresh_mask( hu, thresh_val = 0, mode = 'bigger' )
    # tissue mask 需要再過一次 3x3 dilate
    kernel_dilate = np.ones( ( 3, 3 ), np.uint8 )
    tissue_dilate = cv2.dilate( np.uint8( tissue_mask ), kernel_dilate, iterations = 1 )

    # 從 data_dict 中獲得 body_mask，用於濾除床鋪
    body_np = np.array( body_mask.convert( 'L' ) )
    # 透過 body_np 將 air, bone 中的床鋪移除
    cond_air = to_uint8_255( air_mask * body_np )
    cond_bone = to_uint8_255( bone_mask * body_np )
    cond_tissue = to_uint8_255( tissue_dilate * body_np )

    # 取得 sobel edges:
    # 1. 從已濾除床鋪的 CT 影像開始
    filtered_ct_np = to_uint8_255( np.array( filtered_ct.convert( 'L' ) ) )
    # 2. 獲得初步的 sobel edges
    sobel = gen_sobel( filtered_ct_np )
    # 3. 透過反向操作 bone threshold ，可獲得濾除 bone 的遮罩 => inv_bone
    inv_bone = thresh_mask( hu, thresh_val = 200, mode = 'smaller' )
    # 4. 將 sobel 中的 bone 與空腔濾除
    sobel = sobel * inv_bone * air_mask
    sobel = thresh_mask( image = sobel, thresh_val = 0.05, mode = 'bigger' )
    cond_sobel = to_uint8_255( sobel )
    # 取得類似 semantic segmentation 的 tissue condition
    cond_tissue_old = to_uint8_255( 
        gen_tissue( 
            q_factor = 5, 
            c_factor = 3,
            target_ct = filtered_ct_np, 
            air_mask = air_mask*body_np, 
            bone_mask = bone_mask * body_np, 
            top_k = 5 ) )
    
    cond_wavelet, hh = single_wavelet_2d( image = filtered_ct.convert( 'L' ) )
    if len( cond_wavelet.shape ) >= 3:
        raise ValueError( 'ct-dataset-0905: invalid wavelet condition' )
    cond_wavelet, hh = single_wavelet_2d( image = cond_wavelet )
    cond_wavelet = to_uint8_255( cond_wavelet )

    # 回傳資料
    # 這裡預設回傳的都是 cbct 與其對應的 condition
    if cond_mode == 'air':
        return filtered_ct, post_proc( cond_air )
    elif cond_mode == 'bone':
        return filtered_ct, post_proc( cond_bone )
    elif cond_mode == 'tissue':
        return filtered_ct, post_proc( cond_tissue )
    elif cond_mode == 'body':
        return filtered_ct, body_mask
    elif cond_mode == 'wavelet':
        return filtered_ct, post_proc( cond_wavelet )

class CtDataset0608( datasets.GeneratorBasedBuilder ):
    """CtDataset0608 Images dataset"""

    VERSION = datasets.Version("1.0.0")
    
    def _info( self ):
        features = datasets.Features(
            {
                "image": datasets.Image( decode = True, id = None ),
                "cond": datasets.Image( decode = True, id = None ),
                "prompt": datasets.Value("string")
            }
        )
        # PIL image 可以用 datasets.Image() 進行標記
        # np.ndarray 則可以用 datasets.value()
        # 問題：np.ndarray 是否也可以用 datasets.Image() 標記？
        # => 也許可以，畢竟有 decode 這個 argument 可以選
        # => 在這個 case 上使用 np.ndarray 可能較不合適，因為 ControlNet 下游 pipeline 的預設格式
        #    為 PIL，使用時會報錯
        
        self.dirs = glob( 'dataset/CHEST/CHEST_2021_00*/CT/CT*.dcm' )
        self.dirs.sort()
        self.dirs = self.dirs[ : 400 ] # 只取 400 筆資料，想改請自便

        return datasets.DatasetInfo(
            description = _DESCRIPTION,
            features = features,
            homepage = _HOMEPAGE,
            citation = _CITATION
        )

    def _split_generators(self, dl_manager: DownloadManager):
        
        # 首先獲取所有資料位置
        # ( 應紀錄於 metadata.jsonl 中 )
        # 注意：這裡待補充 !!

        # 決定 condition 模式
        # condition 候選模式：
        # air: 由 hu value 轉換後判定空氣( 空腔 )位置
        # => threshold value 為 -300
        # sobel: 由 pixel_array 透過 sobel operators 得來的邊緣( edges )

        self.mode = 'bone'
        
        print( 'ct-dataset-0905: There are {} samples in total.'.format( len( self.dirs ) ) )
        if self.mode not in [ 'air', 'bone', 'sobel', 'tissue', 'wavelet' ]:
            # 不合規的 condition 都將自動設為 air
            print( "ct-dataset-0905, _split_generators: invalid self.mode: {}, default to \"air\"".format( self.mode ) )
            self.mode = 'air'
        else:
            print( "ct-dataset-0905, _split_generators: mode is set to {}".format( self.mode ) )

        return [
            datasets.SplitGenerator(
                name = datasets.Split.TRAIN,
                gen_kwargs = {
                    "dicom_dir": self.dirs,
                }
            )
        ]
    
    def _generate_examples(self, dicom_dir: List, **kwargs):
        
        idx = -1
        error_cnt = 0
        for path in dicom_dir:
            
            data_dict = self._open_file( dicom_dir = path )
            """
            try:
                data_dict = self._open_file( dicom_dir = path )
            except Exception as e:
                errormsg = str( e )
                print( "error: ", errormsg )
                error_cnt += 1
                continue
            """
            idx += 1
            image = data_dict[ 'image' ].resize( ( 512, 512 ) )
            cond = data_dict[ 'cond' ].resize( ( 512, 512 ) )

            if idx == 0:
                image.save( "controlnet1222_image.png" )
                cond.save( "controlnet_1222_cond.png" )
                print( "inspect dicom name:{}".format( path ) )
            prompt = 'A clean CT image'

            yield idx, {
                        "image" : image,
                        "cond" : cond,
                        "prompt" : prompt
                    }    
    
    def _open_file( self, dicom_dir: str, **kwargs) -> Dict[ str, Image.Image ]:
        # 透過 preprocessing 取得 filtered_ct 與數種 conditions
        
        # image, cond = load_image( data_dir = dicom_dir, cond_mode = self.mode )
        image, cond = load_image_1222( data_dir = dicom_dir, cond_mode = self.mode )
        ret_dict = {
                    "image" : image,
                    "cond" : cond
                }
        
        return ret_dict

if __name__ == '__main__':
    print( 'Dicoms safety check sequence initiating... ' )
    dicoms = glob( 'dataset/ABD/*/CT/CT*.dcm' )
    print( 'There are {} dicom files.'.format( len( dicoms ) ) )

