import glob
import os
import pydicom
from pydicom.dicomdir import DicomDir
from pydicom.dataset import FileDataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from typing import Sequence, List, Type, Tuple, Union, Optional

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

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, 'gray')
    plt.show()
    
    
def bounded(img, bound):
    if img.min() < bound[0] or img.max() > bound[1]:
        return False
    return True


def min_max_normalize(
    img : torch.Tensor, 
    sourceBound : Optional[ Tuple[ int, int ] ] = None, 
    targetBound : Optional[ Tuple[ int, int ] ] = None
    ) -> torch.Tensor:
    """
    min_max_normalize:
    依據給予的 `sourceBound`, `targetBound` 進行縮放

    Args:
    ----------
    img:
    待 normalize 的對象
    ----------
    sourceBound: Optional[ Tuple[ int, int ] ]
    用於正規化 `img` 的參數，以 (min, max) 的格式輸入
    ----------
    targetBound: Optional[ Tuple[ int, int ] ]
    用於讓被正規化的 `img` 在 source domain 中縮放，
    以 (min, max) 的格式輸入

    Return:
    ----------
    torch.Tensor
    正規化後的 `img`
    """
    if sourceBound is None:
        sourceBound = (img.min(), img.max())

    # 若 sourceBound 起始與終點重疊，表示全 0
    if sourceBound[0] == sourceBound[1]:
        if isinstance(img, np.ndarray):
            return np.zeros(img.shape, dtype=np.float32)
        elif isinstance(img, torch.Tensor):
            return torch.where(img > 0, 0)

    # 依據 sourceBound 做 normalization
    img = (img - sourceBound[0])/(sourceBound[1] - sourceBound[0])

    # 若存在 targetBound ，則將樣本依照 targetBound 進行縮放
    if targetBound is not None:
        img = img * (targetBound[1] - targetBound[0]) + targetBound[0]

    return img


def find_mask(
    img : Union[ torch.Tensor, np.ndarray ], 
    width : Optional[ int ] = 1, 
    plot : Optional[ bool ] = False
    ):
    
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    assert len(img.shape) == 2, "Supportive for 2-dims only"
     
    img = img.copy()
    if np.max(img) <= 1:
        img = (img * 255).astype(np.uint8)
    
    img[img != 0] = 255
    img = np.uint8( img )
    cnts, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key = cv2.contourArea)

    if plot:
        img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(img, [c], 0, (255, 255, 255), 3)
        plt.figure(0, figsize=(6,6))
        plt.imshow(img)
        plt.show()
        
    img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(img, [c], 0, (255, 255, 255), width)
    img = img[:, :, 0].astype(np.float32) / 255
    
    return img
    

    
def grow_mask_outward(img, kernel=(5, 5), iterations = 1):
    # https://stackoverflow.com/questions/55948254/scale-contours-up-grow-outward
    kernel = np.ones(kernel, np.uint8)
    img = cv2.dilate(img, kernel, iterations = iterations)
    img = find_mask(img, -1, False)
    return img
    
    
    
def get_mask(img):
    if np.max(img) == 0:
        return np.zeros(img.shape, dtype=np.float32)
    mask = find_mask(img, -1, False)
    mask = grow_mask_outward(mask)
    return mask


def read_dicom( 
    path : str 
    ) -> List[ Type[ FileDataset ] ]:
    """
    read_dicom:
    讀取一批 Dicom 檔的資料

    Args:
    ----------
    path: str
    dicom 資料夾位置，需以 glob 格式傳入。
    e.g. path/to/folder ，這個 folder 之下有數個 .dcm file

    Return:
    ----------
    List[ Type[ FileDataset ] ]
    一個裝有數組 dicom 資料的 list

    """
    g = glob.glob( os.path.join( path, 'CT*.dcm' ) )
    slices = [ pydicom.read_file(s) for s in g ]
    
    slices : List[ Type[ FileDataset ] ]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices


def hu_clip_tensor(
    img : torch.Tensor, 
    sourceBound : Tuple[ int, int ], 
    targetBound : Optional[ Tuple[ int, int ] ] = None, 
    min_max_norm : Optional[ bool ] = True
    ) -> torch.Tensor:
    """
    hu_clip_tensor:
    對於傳入的樣本 `img` 針對給予的 `sourceBound` 範圍進行 clipping & normalization。
    先基於 `sourceBound` 進行 clipping，
    在基於 `sourceBound` 與 `targetBound` 進行 normalization

    Args:
    ----------
    img: torch.Tensor
    待正規化的樣本
    ----------
    sourceBound: Tuple[ int, int ]
    用於對影像做正規化
    ---------
    targetBound: Optional[ Tuple[ int, int ] ]
    用於對影像進一步縮放
    ----------
    min_max_norm: Optional[ bool ]
    是否需要對影像進行正規化，預設為需要

    Return:
    ----------
    torch.Tensor
    clipping 後的樣本
    """

    lower = sourceBound[0]
    upper = sourceBound[1]
    img = torch.where(img < lower, lower, img)
    img = torch.where(img > upper, upper, img)
    if min_max_norm:
        img = min_max_normalize(img, sourceBound, targetBound)
        
    return img


def hu_clip(
    img, 
    sourceBound, 
    targetBound = None, 
    min_max_norm = True, 
    zipped = False
    ):

    lower = sourceBound[0]
    upper = sourceBound[1]
    img = np.where(img < lower, lower, img)
    img = np.where(img > upper, upper, img)

    if min_max_norm:
        img = min_max_normalize(img, sourceBound, targetBound)
    
    if zipped:
        img = (img*255).astype(np.uint8).astype(np.float32) / 255
        
    return img


def hu_window(scan, window_level=40, window_width=80, min_max_norm=True):
    scan = scan.pixel_array.copy()
    window = [window_level-window_width//2, window_width//2-window_level]
    
    scan = np.where(scan < window[0], window[0], scan)
    scan = np.where(scan > window[1], window[1], scan)
    
    if min_max_norm:
        scan = min_max_normalize(scan,(upper, lower))

    return scan


def show_raw_pixel(slices):
    #讀出像素值並且儲存成numpy的格式
    image = hu_window(slices, window_level=0, window_width=1000,  show_hist=False)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()
    
    
    
# return  a region where cbct images aren't all black
def valid_slices( cbcts : List[ FileDataset ] ) -> Tuple[ int, int ]:
    """
    回傳一個區間，區間內不存在全黑的 dicom 檔
    """
    found_start = False
    start = 0
    end = len( cbcts ) - 1
    
    # iterate over cbct slices, and find which regions aren't all black (-1000)
    for idx, sli in enumerate( cbcts ):
        image = sli.pixel_array
        
        if not found_start and len( np.unique( image ) ) != 1:
            start = idx
            found_start = True
                                   
        elif found_start and len( np.unique( image ) ) == 1:
            end = idx
            break
        
    return start, end

