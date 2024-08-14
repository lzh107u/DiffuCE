# 2023-07-31

import numpy as np
import PIL.Image as Image
from skimage.transform import radon
from skimage.transform import iradon
from typing import Tuple, List, Union, Sequence
from dicom_utils import dicom_pipeline
from glob import glob
from training_utils import write_file

#### 將輸入的 np.ndarray 正規化到 0 ~ 1 之間 ####
def normalize( arr: np.ndarray ) -> np.ndarray:
    arr = arr - np.min( arr )
    arr = arr / np.max( arr )
    return arr
#### 將輸入的 np.ndarray 以 np.uint8 格式正規化到 0-255 之間 ####
def to_uint8_255( arr: np.ndarray ) -> np.ndarray:
    arr = normalize( arr )
    arr = np.uint8( arr * 255 )
    return arr

def to_sinogram( target: np.ndarray ) -> np.ndarray:
    """
    to_sinogram:
    將 image 透過 radon transform 投影到 sinogram domain 上

    Args:
    ----------
    target: np.ndarray
    待轉換的 image

    Return:
    ----------
    sinogram: np.ndarray
    輸入 image 的 sinogram，長寬為投影軸長x投影數量，即 725x512
    
    """
    theta = np.linspace( 0., 180., 512, endpoint=False )
    sinogram = radon( target, theta = theta, circle = False )
    return sinogram

def sino_reconstruct( sino: np.ndarray ) -> np.ndarray:
    """
    sino_reconstruct:
    將 sinogram 透過 inverse radon transform 轉換回 image domain
    注意：
    由於 radon transform 的特性，512x512 的影像在轉換成 sinogram 後，會變成：
        投影數 x 原圖對角線長
    如此才不會在轉換後喪失部分外圍資訊。
    目前的設計中，投影數為 512 ，對角線長度為 512*√2 = 725，
    故輸入的大小應該是 725x512

    Args:
    ----------
    sino: np.ndarray
    待轉換的 sinogram

    Return:
    ----------
    reconstruction_fbp: np.ndarray
    將輸入的 sinogram 重新投影回 image domain 的結果

    """
    theta = np.linspace( 0., 180., 512, endpoint=False )
    reconstruction_fbp = iradon( sino, theta=theta, filter_name='hann', output_size = 512 )
    return reconstruction_fbp

def drop_periodically( sino_shape: Tuple[ int, int ], drop_area_cnt: Union[ int, float ] = 10 ) -> np.ndarray:
    """
    drop_periodically:
    週期性地丟棄/保留資訊

    Args:
    ----------
    sino_shape: Tuple[ int, int ]
    輸出遮罩的長寬，若這裡的 shape 與待處理的 array 不一致，會讓計算出錯
    ----------
    drop_area_cnt: int
    丟棄週期，單位為每 n 個 view 留下 1 個，保留資訊量為 1/n

    Return:
    ----------
    mask_template: np.ndarray
    資訊丟棄遮罩，dropping mask

    """

    # define sparsity
    # 決定間隔多少才有一個 view
    
    if drop_area_cnt > 1:
        # 丟棄週期大於 1 ，表示丟掉的比留下的多，每 D.A.C. 個 column 才留一個 view
        mask_template = np.zeros( sino_shape[ 1 ], dtype = np.float64 )
        drop_area_cnt = round( drop_area_cnt ) # 用在 indice 上需要是整數
        mask_template[ :: drop_area_cnt ] = 1.0
    elif drop_area_cnt < 1 and drop_area_cnt > 0:
        # 丟棄週期小於 1 ，表示留下的比丟掉的多，每 D.A.C. 個 column 才丟掉一個 view
        mask_template = np.ones( sino_shape[ 1 ], dtype = np.float64 )
        drop_area_cnt = int( 1 / drop_area_cnt )
        mask_template[ :: drop_area_cnt ] = 0.0
    else:
        # 其餘 case 皆當作不 drop
        mask_template = np.ones( sino_shape[ 1 ], dtype = np.float64 )
    
    # 將 1 維的 mask pattern 垂直堆疊起來
    mask_template = np.vstack( [ mask_template, mask_template ] )
    while mask_template.shape[ 0 ] < sino_shape[ 0 ]:
        mask_template = np.vstack( [ mask_template, mask_template ] )
    mask_template = mask_template[ : sino_shape[ 0 ], : ]

    return mask_template

def drop_randomly( sino_shape: Tuple[ int, int ], p: float = 0.4 ) -> np.ndarray:
    """
    drop_randomly:
    隨機地丟棄資訊，透過給定的 p 決定保留的資訊比例

    Args:
    ----------
    sino_shape: Tuple[ int, int ]
    輸出遮罩的長寬，若這裡的 shape 與待處理的 array 不一致，會讓計算出錯
    ----------
    p: float
    丟棄資訊的比例，值域為 [ 0, 1 ]

    Return:
    ----------
    mask_template: np.ndarray
    資訊丟棄遮罩，dropping mask

    """
    # 首先以 normal distribution 生成一個 0 ~ 1 的 normal distribution 
    # 藉由 p 作為 threshold ，大於 p 的位置才採用，此時被選上的 view 比例應趨近 p 
    mask_template = np.zeros( sino_shape[ 1 ], dtype = np.float64 )
    random_map = np.random.rand( sino_shape[ 1 ] )
    for col in range( sino_shape[ 1 ] ):
        if random_map[ col ] > p:
            mask_template[ col ] = 1.0

    # 將 1 維的 mask pattern 垂直堆疊起來
    mask_template = np.vstack( [ mask_template, mask_template ] )
    while mask_template.shape[ 0 ] < sino_shape[ 0 ]:
        mask_template = np.vstack( [ mask_template, mask_template ] )
    mask_template = mask_template[ : sino_shape[ 0 ], : ]
    return mask_template

def sinogram_downsampling( sino_orig: np.ndarray, 
                        flag: str = 'random', 
                        p: float = 0.4, 
                        period: Union[ int, float ] = 10 ) -> np.ndarray:
    """
    sinogram_downsampling: 
    在 sinogram domain 上丟棄資訊，人工地製造 sparse-view CT

    Args:
    ----------
    sino_orig: np.ndarray
    待處理的 sinogram ，可以當成是 full-view CT
    ----------
    flag: str
    決定 dropping 的模式，分成：
        "random" : 隨機丟棄
        "period" : 週期丟棄
    預設為 "random"
    ----------
    p: float
    在隨機模式下，決定保留資料的比例
    預設為 0.4 ，保留 40% 資料的意思
    ----------
    period: Union[ int, float ]
    在週期模式下，決定保留資料的比例
    預設為 10 ，保留 1/10 資料的意思
    Return:
    ----------
    masked_sinogram: np.ndarray
    已經 dropped 的 sinogram ，此時已經是 sparse-view sinogram
    
    """
    if flag == 'random':
        # 隨機地 drop
        mask4sinogram = drop_randomly( sino_shape = sino_orig.shape, p = p )
    elif flag == 'period':
        # 週期性地 drop
        mask4sinogram = drop_periodically( sino_shape = sino_orig.shape, drop_area_cnt = period )
    else:
        # 將回傳完整 sinogram 原圖
        mask4sinogram = drop_periodically( sino_shape = sino_orig.shape, drop_area_cnt = 1 )
    
    masked_sinogram = sino_orig * mask4sinogram
    # imshow( [ 'origin', 'masked-{}'.format( flag ) ], [ sino_orig, masked_sinogram ] )
    return masked_sinogram

def cbct_pipeline( sino_fw: np.ndarray ) -> np.ndarray:
    """
    cbct_pipeline:
    輸入 full-view sinogram，輸出 dropped sparse-view sinogram

    Args:
    ----------
    sino_fw: np.ndarray
    待處理的 full-view sinogram

    Return:
    ----------
    cbct_sino: np.ndarray
    已經丟棄部分資訊的 sparse-view sinogram 

    """
    sino_dwsp = sinogram_downsampling( sino_orig = sino_fw )
    cbct_imgdomain = sino_reconstruct( sino = sino_dwsp )
    cbct_sino = to_sinogram( cbct_imgdomain )
    return cbct_sino

def sino_pipeline( images: Sequence[ Image.Image ] ) -> List[ np.ndarray ]:
    """
    sino_pipeline:
    輸入一系列 images，將它們全部透過 radon transform 投影至 sinogram domain

    Args:
    ----------
    images: Sequence[ Image.Image ]
    一系列待轉換的 image

    Return:
    ----------
    result: List[ np.ndarray ]
    一系列 sinogram

    """
    result = []
    for idx in range( len( images ) ):
        arr = np.array( images[ idx ].convert("L") )
        sinogram = to_sinogram( arr )
        result.append( normalize( sinogram ) )
    return result

def sino_combine( sinos: Sequence[ np.ndarray ], weights: Sequence[ float ] = [ 1 ], flag_norm: bool = True ) -> np.ndarray:
    """
    sino_combine:
    將輸入的多組 sinogram 合併

    Args:
    ----------
    sinos: Sequence[ np.ndarray ]
    待合併的一系列 sinograms
    ----------
    weights: Sequence[ float ]

    ----------
    flag_norm: bool
    是否需要正規化至 0-1 ，預設為開啟

    Return:
    ----------
    template: np.ndarray
    回傳合併後的 sinogram

    """
    template = np.zeros( sinos[ 0 ].shape )
    for idx, sino in enumerate( sinos ):
        # w 是每一組 sinogram 的權重，若在賦值時出現任何錯誤，都將把 w 設為 1
        try:
            w = weights[ idx ]
        except:
            w = 1
        
        template = template + w*sino
    
    if flag_norm is True:
        return normalize( template )
    return template

def recon_pipeline( sinograms: Union[ Sequence[ Image.Image ], Sequence[ np.ndarray ] ], 
                flag_pil: bool = False, 
                weights: Sequence[ float ] = None ) -> List[ Union[ np.ndarray, Image.Image] ]:
    """
    recon_pipeline:
    輸入一系列 sinogram ，將它們全部透過 inverse radon transform 投影回 image domain

    Args:
    ----------
    sinograms: Union[ Sequence[ Image.Image ], Sequence[ np.ndarray ] ]
    一系列 sinogram，每張長寬都須符合當前設定，即：
        投影數量：512 views
        投影軸長：725
    注意：
    轉換將在 np.ndarray 格式下進行，若輸入格式為 Image.Image 將自動轉檔，不支援其他格式
    ----------
    flag_pil: bool
    是否輸出為 PIL.Image.Image
    ----------
    weights: Sequence[ float ]
    一系列權重，用於對輸入的 sinograms 進行加權平均
    預設為 None ，則不進行加權平均

    Return:
    ----------
    result: List[ np.ndarray | Image.Image ]
    一系列轉換至 image domain 的影像 

    """
    result = []

    if weights is not None:
        for arr in sinograms:
            if not isinstance( arr, np.ndarray ):
                arr = np.array( arr.convert( "L" ).resize( ( 512, 725 ) ) )
            result.append( arr )
        result = sino_combine( result, weights )
        result = sino_reconstruct( sino = result )
        if flag_pil is True:
            return [ Image.fromarray( to_uint8_255( result ) ) ]
        else:
            return [ to_uint8_255( result ) ]

    result = []
    for arr in sinograms:
        if not isinstance( arr, np.ndarray ):
            arr = np.array( arr.convert( "L" ) )
        recon = sino_reconstruct( sino = arr )

        if flag_pil is True:
            result.append( Image.fromarray( to_uint8_255( recon ) ) )
        else:
            result.append( to_uint8_255( recon ) )

    return result

def ct_to_cbct_image(   
    ct_images : Sequence[ Image.Image ],
    masks : Union[ Sequence[ Image.Image ], None ] = None 
    ) -> Sequence[ Image.Image ]:
    """
    ct_to_cbct_image:
    輸入一系列 CT( full-view ) ，輸出一系列 CBCT( sparse-view )

    Args:
    ----------
    ct_images: Sequence[ Image.Image ]
    一系列待轉換的原始 CT 
    ----------
    masks: Sequence[ Image.Image ] | None, optional
    一系列 body mask ，用於遮掉多餘的 Artifact

    Return:
    ----------
    Sequence[ Image.Image ]
    一系列轉換完畢的 CBCT( sparse-view )

    註：
    Radon Transform 需要時間，建議透過前處理執行轉換
    iir gpu server( titan2 ): 4 sec per image
    my laptop: 9 sec per image
    """

    ct_sinograms = sino_pipeline( ct_images )
    if masks is not None and len( masks ) != len( ct_images ):
        print( 'ct_to_cbct_image: warning, the number of masks does not match with `ct_images`' )
        print( 'expect {}, got {}'.format( len( ct_images ), len( masks ) ) )
    
    def to_sparse_view( sino: np.ndarray, idx: int ) -> Image.Image:
        """
        to_sparse_view:
        輸入一張 sinogram( full-view )，輸出一張 sparse-view CT image

        Args:
        ----------
        sino: np.ndarray
        full-view sinogram ，就是原始 CT
        ----------
        idx: int
        用於存取特定的 mask

        Return:
        ----------
        Image.Image:
        一張原始 CT 的稀疏版本，可視為 CBCT

        """
        sino = sinogram_downsampling( sino_orig = sino )
        img = sino_reconstruct( sino = sino ) # 這裡還是 np.ndarray
        if masks is not None and len( masks ) > 0:
            # 若 masks 有被傳入，
            mask = masks[ idx % len( masks ) ]
            mask = np.array( mask.convert( 'L' ) )
            mask = np.where( mask > 0, 1, 0 )
            img = img * mask
            img = to_uint8_255( arr = img )
        img = Image.fromarray( obj = img ).resize( ( 512, 512 ) ).convert( 'RGB' )
        return img

    cbct_sinogram = [ to_sparse_view( sino = sinogram, idx = idx ) for idx, sinogram in enumerate( ct_sinograms ) ]
    
    return cbct_sinogram

def demo():
    
    dicoms : List[ str ] = glob( 'dataset/CHEST/*/CT/CT*.dcm' )

    images : List[ Image.Image ] = dicom_pipeline( ds_dir = dicoms[ 258 ] )
    print("Converting, please wait for few seconds...")
    cbct : Image.Image = ct_to_cbct_image( ct_images = [ images[ 0 ] ], masks = [ images[ 5 ] ] )[ 0 ]
    print("Done!!")  

    @write_file
    def save_figure( file_path : str, image: Image.Image ) -> None:
        image.save( file_path )

    save_figure( file_path = 'ct0809.png', image = images[ 0 ] )
    save_figure( file_path = 'cbct0809.png', image = cbct )
    
    return 

if __name__ == '__main__':
    demo()
