import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import PIL.Image as Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import ConcatDataset

from .utils import hu_clip, read_dicom, valid_slices, min_max_normalize, transform_to_hu
from .augmentations.base import refine_mask
from .augmentations.denoise_mask import DenoiseMask
from .augmentations.window_mask import AirBoneMask

from .augmentations.mask import get_mask
from .augmentations.air_bone_mask import get_air_bone_mask

from typing import Type, List, Tuple, Any, Sequence, Iterable, Optional, Union
from pydicom.dicomdir import DicomDir
from pydicom import dcmread
import pickle

class DicomDataset(BaseDataset):
    """
    Args:
        path (str): path to dataset
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(
        self, 
        cbct_path : str, 
        ct_path : str, 
        ditch : int = 3,          
        intensity_aug : nn.Module = None, 
        geometry_aug : nn.Module = None, 
        identity : bool = False, 
        electron : bool = False, 
        position : str = "pelvic", 
        g_coord : bool = False, 
        l_coord : bool = False ) -> None:
        """
        Args:
        ----------
        cbct_path: str
        ----------
        ct_path: str
        ----------
        ditch: int
        有效區間內拋棄張數
        ----------
        identity: bool
        是否將 x, y 設置成同一種東西，具體差異參見 __getitem__
        ----------
        electron: bool
        似乎是指資料是否為 dicom 檔的 pixel_array raw data ，若為 `False` 會認為是正規化後的值域
        ----------
         g_coord: bool
         是否需要 global coordinates，具體差異為是否在 __init__ 中回傳 `p_encoding`
         ----------
         l_coord: bool
         是否需要 local coordinates，具體差異為是否在 __init__ 中回傳 `local_encoding`

        """

        # read cbct and ct
        assert cbct_path.split("/")[-2].split("_")[-1] == ct_path.split("/")[-2].split("_")[-1]     
        self.patient_id = cbct_path.split("/")[-2].split("_")[-1]
        
        # 根據傳入的 path 讀取同一個 patient 的 CT 與 CBCT 
        cbct_slices = read_dicom( cbct_path )
        ct_slices = read_dicom( ct_path )

        region = valid_slices(cbct_slices)
        # ditch first and last 3
        # 拋棄區間內前後三張影像
        self.xs = cbct_slices[region[0] + ditch: region[1] - ditch]
        self.ys = ct_slices[region[0] + ditch: region[1] - ditch]

        if len( self.xs ) != len( self.ys ):
            print( 'Caution: xs({}) != ys({}) in patient_id: {}'.format( len( self.xs ), len( self.ys ), self.patient_id ) )
            self.xs = self.xs[ : len( self.ys ) ]
            print( 'Resize xs to {}'.format( len( self.ys ) ) )

        encoding = []
        length = (region[1] - ditch) - (region[0] + ditch)
        quotient, remainder = length // 5, length % 5

        for i in range(5):
            cnt = quotient
            if remainder > 0:
                cnt = cnt + 1
                remainder = remainder - 1
            encoding = encoding + [i for _ in range(cnt)]
        self.encoding = encoding

        self.position = position
        self.identity = identity
        self.electron = electron
        self.g_coord = g_coord
        self.l_coord = l_coord
        self.intensity_aug = intensity_aug
        self.geometry_aug = geometry_aug

        self.x_norm = (0, 1)
        self.y_norm = (0, 1)
        if electron:
            self.y_norm = (-1015.1, 1005.8)
            if position == "pelvic" or position == "abdomen":
                self.x_norm = (-1052.1, 1053.4)
            elif position == "chest":
                self.x_norm = (-1059.6, 1067.2)
            elif position == "headneck":
                self.x_norm = (-1068.8, 1073.5)
            else:
                assert False, "Position: pelvic, abdomen, chest, and headneck"

                
        self.p_encoding = 0
        if position == "headneck":
            self.p_encoding = 0
        elif position == "chest":
            self.p_encoding = 1
        elif position == "abdomen":
            self.p_encoding = 2
        elif position == "pelvic":
            self.p_encoding = 3
        else:
            assert False, "Position: pelvic, abdomen, chest, and headneck"      
                
                
    def __getitem__( self, i : int ) -> Sequence[ torch.Tensor ]:

        # read img
        # 這裡讀入 dicom raw data
        # 注意值域是自然數，且形態為 np.int16
        x = self.xs[i].pixel_array.copy()
        y = self.ys[i].pixel_array.copy()

        x_hu = transform_to_hu( self.xs[ i ], x )
        y_hu = transform_to_hu( self.ys[ i ], y )
        encoding = self.encoding[i]
        p_encoding = self.p_encoding
        
       ############################
        # Data denoising
        ###########################  
        denoise_bound = (-500, -499)
        if self.electron:
            y = (y - self.y_norm[0])/self.y_norm[1]
            x = (x - self.x_norm[0])/self.x_norm[1]
            denoise_bound = (0.4, 0.5)
            
        mask_x = DenoiseMask(bound=denoise_bound, always_apply=True)(image=x_hu)["image"]
        mask_y = DenoiseMask(bound=denoise_bound, always_apply=True)(image=y_hu)["image"]

        view_bound = (-500, 500)
        if self.electron:
            view_bound = (0.5, 1.5)

        x = x * mask_x
        y = y * mask_y

        x = transform_to_hu( self.xs[ i ], x )
        y = transform_to_hu( self.ys[ i ], y )
        
        x = hu_clip(x, view_bound, None, False )
        y = hu_clip(y, view_bound, None, False )
        
       ############################
        # Get air bone mask
        ###########################          
        air_bound = ( -301, -300 )
        bone_bound = ( 200, 201 )
        if self.electron:
            air_bound = (0.5, 0.5009)
            bone_bound = (1.2, 1.2009)
        sample = AirBoneMask(bound=view_bound, air_bound=air_bound, bone_bound=bone_bound, always_apply=True)(image=x)["image"]
        air_x, bone_x = sample[0, :, :], sample[1, :, :]
        
        sample = AirBoneMask(bound=view_bound, air_bound=air_bound, bone_bound=bone_bound, always_apply=True)(image=y)["image"]
        air_y, bone_y = sample[0, :, :], sample[1, :, :]
        
        bone_x = refine_mask(bone_x, bone_y)
        
        if self.geometry_aug:
            sample = self.geometry_aug(image=x, image0=y, image1=air_x, image2=bone_x, image3=air_y, image4=bone_y)
            x, y, air_x, bone_x, air_y, bone_y = sample["image"], sample["image0"], \
                                                    sample["image1"], sample["image2"], \
                                                    sample["image3"], sample["image4"]
                    
        k = bone_x
        if self.intensity_aug:
            sample = self.intensity_aug(image=x, image0=y, image1=air_x, image2=bone_x, image3=air_y, image4=bone_y)
            x, y, air_x, bone_x, air_y, bone_y = sample["image"], sample["image0"], \
                                                    sample["image1"], sample["image2"], \
                                                    sample["image3"], sample["image4"]
        
        x = np.expand_dims(x, 0).astype(np.float32)
        y = np.expand_dims(y, 0).astype(np.float32)
        air_x = np.expand_dims(air_x, 0).astype(np.float32)
        bone_x = np.expand_dims(bone_x, 0).astype(np.float32)
        air_y = np.expand_dims(air_y, 0).astype(np.float32)
        bone_y = np.expand_dims(bone_y, 0).astype(np.float32)
    
        encoding = np.ones(x.shape, dtype=np.float32) * encoding
        p_encoding = np.ones(x.shape, dtype=np.float32) * p_encoding
        c, w, h = x.shape
        local_encoding = self.make_grid2d(w, h)

        if self.identity:
            x = y
            air_x = air_y
            bone_x = bone_y
        
        
        k = k - np.min( k )
        k = k / np.max( k )
        k = k * 255
        k = np.uint8( k )

        # img = Image.fromarray( k, 'L' )
        # img.save( 'cyclegan_dataset.png')


        if self.g_coord and self.l_coord:
            return x, y, air_x, bone_x, air_y, bone_y, p_encoding, local_encoding
        elif self.g_coord and not self.l_coord:
            return x, y, air_x, bone_x, air_y, bone_y, p_encoding
        elif not self.g_coord and self.l_coord:
            return x, y, air_x, bone_x, air_y, bone_y, local_encoding    

        
        return x, y, air_x, bone_x, air_y, bone_y

    def __len__(self):
        return len(self.xs)
    
    
    def patientID(self):
        return self.patient_id
    

    def make_grid2d(self, height, width):
        h, w = height, width
        grid_x, grid_y = np.meshgrid(np.arange(0, h), np.arange(0, w))
        grid_x = 2 * grid_x / max(float(w) - 1., 1.) - 1.
        grid_y = 2 * grid_y / max(float(h) - 1., 1.) - 1.
        grid = np.stack((grid_x, grid_y), 0)

        return grid
    
class PickleDataset( BaseDataset ):

    def __init__(
        self,
        pickle_path : str,

    ) -> None:
        


        return 
    
def DicomsDataset(
    path : str, 
    geometry_aug : nn.Module = None, 
    intensity_aug : nn.Module = None, 
    identity : bool = False, 
    electron : bool = False, 
    position : str = "pelvic", 
    g_coord : bool = False, 
    l_coord : bool = False ):
    """
    DicomsDataset:
    透過將數個 DicomDataset 整合成一個 ConcatDataset ，將所有 dicom 檔案合併在一起。
    傳入以 Iterable 包裝的 Datasets ，在內部是以 List[ Dataset ] 進行儲存；
    indexing 時是以`累積`的方式進行跨 subset 的存取，
    若傳入的 index 大於累積到 set_k 的資料總量時，就會存取 set_k+1 內的資料。
    
    參閱：https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset

    Args:
    ----------
    path: str
    指向 dicom 資料夾的位置，這裡要使用 glob 函數庫的慣用格式
    e.g. path/to/folder/*.dcm
    ----------
    geometry_aug: nn.Module
    用於對資料進行擴增，請傳入 torchvision.transform 的函數指標
    e.g. transform.RandomFlip, transform.CenterCrop
    ----------
    intensity_aug: nn.Module
    概念與 `geometry_aug` 相近
    ----------
    electron: bool
    default to `False`
    ----------
    position: str
    標記是哪個 Folder
    ----------
    g_coord: bool
    default to `False`
    ----------
    l_coord: bool
    default to `False`

    Return:
    ----------
    ConcatDataset( Type[ Dataset ] )

    """
    paths = sorted( glob( path ) )
    datasets = []
    
    # read cbct and ct
    for i in range(0, len(paths), 2):
        scans : Type[ BaseDataset ] = DicomDataset(
            cbct_path = paths[i], 
            ct_path = paths[ i + 1 ], 
            ditch = 3, 
            geometry_aug = geometry_aug, 
            intensity_aug = intensity_aug, 
            identity = identity, 
            electron = electron, 
            position = position, 
            g_coord = g_coord, 
            l_coord = l_coord )
        
        datasets = datasets + [ scans ]
        
    datasets : Iterable[ BaseDataset ]
    datasets = ConcatDataset( datasets )
    return datasets

    

    
class DicomSegmentDataset(BaseDataset):
    """
    Args:
        path (str): path to dataset
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(self, cbct_path, ct_path, ditch=3, segment=8, 
                 intensity_aug=None, geometry_aug=None, 
                 identity=False, electron=False, position="pelvic", g_coord=False, l_coord=False):

        # read cbct and ct
        assert cbct_path.split("/")[-1].split("_")[0] == ct_path.split("/")[-1].split("_")[0]     
        self.patient_id = cbct_path.split("/")[-1].split("_")[0] 
        
        cbct_slices = read_dicom(cbct_path)
        ct_slices = read_dicom(ct_path)

        region = valid_slices(cbct_slices)
        # ditch first and last 3
        self.xs = cbct_slices[region[0] + ditch: region[1] - ditch]
        self.ys = ct_slices[region[0] + ditch: region[1] - ditch]
        
        encoding = []
        length = (region[1] - ditch) - (region[0] + ditch)
        quotient, remainder = length // 5, length % 5

        for i in range(5):
            cnt = quotient
            if remainder > 0:
                cnt = cnt + 1
                remainder = remainder - 1
            encoding = encoding + [i for _ in range(cnt)]
        self.encoding = encoding
        
        self.position = position
        self.segment = segment
        self.identity = identity
        self.electron = electron
        self.g_coord = g_coord
        self.l_coord = l_coord
        self.intensity_aug = intensity_aug
        self.geometry_aug = geometry_aug

        self.x_norm = (0, 1)
        self.y_norm = (0, 1)
        if electron:
            self.y_norm = (-1015.1, 1005.8)
            if position == "pelvic" or position == "abdomen":
                self.x_norm = (-1052.1, 1053.4)
            elif position == "chest":
                self.x_norm = (-1059.6, 1067.2)
            elif position == "headneck":
                self.x_norm = (-1068.8, 1073.5)
            else:
                assert False, "Position: pelvic, abdomen, chest, and headneck"

                
        self.p_encoding = 0
        if position == "headneck":
            self.p_encoding = 0
        elif position == "chest":
            self.p_encoding = 1
        elif position == "abdomen":
            self.p_encoding = 2
        elif position == "pelvic":
            self.p_encoding = 3
        else:
            assert False, "Position: pelvic, abdomen, chest, and headneck"      
                
                
    def __getitem__(self, idx):

        # read img
        xs = []
        air_xs = []
        bone_xs = []
        ys = []
        air_ys = []
        bone_ys = []
        encodings = []
        
        index = np.array(range(self.segment)) - self.segment // 2

        for s in index:
            i = s + idx
            
            if i < 0 or i >= self.__len__():
                x = np.zeros((512, 512), dtype=np.float32)
                y = np.zeros((512, 512), dtype=np.float32)
                encoding = 0 if i<0 else 4
                air_x = np.zeros((512, 512), dtype=np.float32)
                bone_x = np.zeros((512, 512), dtype=np.float32)
                air_y = np.zeros((512, 512), dtype=np.float32)
                bone_y = np.zeros((512, 512), dtype=np.float32)       
                
            else:
                x = self.xs[i].pixel_array.copy()
                y = self.ys[i].pixel_array.copy()
                encoding = self.encoding[i]
               ############################
                # Data denoising
                ###########################  
                denoise_bound = (-512, -257)
                if self.electron:
                    y = (y - self.y_norm[0])/self.y_norm[1]
                    x = (x - self.x_norm[0])/self.x_norm[1]
                    denoise_bound = (0.4, 0.5)

                mask_x = DenoiseMask(bound=denoise_bound, always_apply=True)(image=x)["image"]
                mask_y = DenoiseMask(bound=denoise_bound, always_apply=True)(image=y)["image"]

                view_bound = (-500, 500)
                if self.electron:
                    view_bound = (0.5, 1.5)
                x = hu_clip(x, view_bound, None, True)
                y = hu_clip(y, view_bound, None, True)

                x = x * mask_x
                y = y * mask_y

               ############################
                # Get air bone mask
                ###########################          
                air_bound =(-500, -499) # -500, -300
                bone_bound = (255, 256) # 300, 500
                if self.electron:
                    air_bound = (0.5, 0.5009)
                    bone_bound = (1.2, 1.2009)

                sample = AirBoneMask(bound=view_bound, air_bound=air_bound, bone_bound=bone_bound, always_apply=True)(image=x)["image"]
                air_x, bone_x = sample[0, :, :], sample[1, :, :]
                sample = AirBoneMask(bound=view_bound, air_bound=air_bound, bone_bound=bone_bound, always_apply=True)(image=y)["image"]
                air_y, bone_y = sample[0, :, :], sample[1, :, :]

                bone_x = refine_mask(bone_x, bone_y)

            xs += [x]
            air_xs += [air_x]
            bone_xs += [bone_x]
            ys += [y]
            air_ys += [air_y]
            bone_ys += [bone_y]
            encodings += [encoding]
        
        xs = np.stack(xs, axis=-1)
        air_xs = np.stack(air_xs, axis=-1)
        bone_xs = np.stack(bone_xs, axis=-1)
        ys = np.stack(ys, axis=-1)
        air_ys = np.stack(air_ys, axis=-1)
        bone_ys = np.stack(bone_ys, axis=-1)

        
        if self.geometry_aug:
            sample = self.geometry_aug(image=xs, image0=ys, image1=air_xs, image2=bone_xs, image3=air_ys, image4=bone_ys)
            xs, ys, air_xs, bone_xs, air_ys, bone_ys = sample["image"], sample["image0"], \
                                                                                                                        sample["image1"], sample["image2"], \
                                                                                                                        sample["image3"], sample["image4"]
                    
            
        if self.intensity_aug:
            sample = self.intensity_aug(image=xs, image0=ys, image1=air_xs, image2=bone_xs, image3=air_ys, image4=bone_ys)
            xs, ys, air_xs, bone_xs, air_ys, bone_ys = sample["image"], sample["image0"], \
                                                                                                                        sample["image1"], sample["image2"], \
                                                                                                                        sample["image3"], sample["image4"]
        
        encodings = np.ones(xs.shape, dtype=np.float32) * encodings
        encodings = np.expand_dims(np.moveaxis(encodings, -1, 0), 1)
        xs = np.expand_dims(np.moveaxis(xs, -1, 0), 1)
        ys = np.expand_dims(np.moveaxis(ys, -1, 0), 1)
        air_xs = np.expand_dims(np.moveaxis(air_xs, -1, 0), 1)
        bone_xs = np.expand_dims(np.moveaxis(bone_xs, -1, 0), 1)
        air_ys = np.expand_dims(np.moveaxis(air_ys, -1, 0), 1)
        bone_ys = np.expand_dims(np.moveaxis(bone_ys, -1, 0), 1)
        
        if self.identity:
            xs = ys
            air_xs = air_ys
            bone_xs = bone_ys
        
        if self.g_coord:
            return xs, ys, air_xs, bone_xs, air_ys, bone_ys, encodings
        return xs, ys, air_xs, bone_xs, air_ys, bone_ys
    
        
    def __len__(self):
        return len(self.xs)
    
    
    def patientID(self):
        return self.patient_id
    

    def make_grid2d(self, height, width):
        h, w = height, width
        grid_x, grid_y = np.meshgrid(np.arange(0, h), np.arange(0, w))
        grid_x = 2 * grid_x / max(float(w) - 1., 1.) - 1.
        grid_y = 2 * grid_y / max(float(h) - 1., 1.) - 1.
        grid = np.stack((grid_x, grid_y), 0)

        return grid
    

def DicomsSegmentDataset(path, geometry_aug=None, intensity_aug=None, 
                         identity=False, electron=False, position="pelvic", segment=8, g_coord=False, l_coord=False):
        paths = sorted(glob(path))
        
        datasets = []
        # read cbct and ct
        for i in range(0, len(paths), 2):
            scans = DicomSegmentDataset(cbct_path=paths[i+1], ct_path=paths[i], ditch=3, segment=segment, 
                                 geometry_aug=geometry_aug, intensity_aug=intensity_aug, 
                                 identity=identity, electron=electron, position=position, g_coord=g_coord, l_coord=l_coord)
            datasets = datasets + [scans]
            
        datasets = ConcatDataset(datasets)
        return datasets

SET_RANGE_LIST = [
    ( 8, 51 ),
    ( 0, 48 ),
    ( 14, 65 ),
    ( 18, 70 ),
    ( 9, 62 ),
    None,
    ( 12, 59 ),
    None,
    None,
    ( 21, 63 ),
]

def path_prepare_0915(
    path_to_dataset : str # 'dataset/ABD/*'
) -> List[ str ]:
    """
    path_prepare_0915:
    產出 training set 的 dicom 集合

    Args:
    ----------
    path_to_dataset: str
    請傳入 ABD 資料集位置，預設為 "dataset/ABD/*"

    Return:
    ----------
    List[ str ]
    dicom 位置集合

    """
    folders = glob( path_to_dataset )
    folders.sort()
    folders = folders[ : 10 ]
    all_dicoms = []

    for idx, folder in enumerate( folders ):
        dicoms = glob( folder + '/CT/CT*.dcm' )
        dicoms.sort()
        if SET_RANGE_LIST[ idx ] is None:
            continue
        
        dicoms = dicoms[ SET_RANGE_LIST[ idx ][ 0 ] : SET_RANGE_LIST[ idx ][ 1 ] + 1 ]
        all_dicoms += dicoms
    
    return all_dicoms

def gan_preprocessing( 
    dicom : Union[ DicomDir, str ],
    pkl : Union[ int, str ] = None,
    path_to_dataset : Optional[ str ] = "dataset/ABD/*",
    geometry_aug : Optional[ bool ] = False,
    intensity_aug : Optional[ bool ] = False,
    func_geometry : Optional[ Type[ nn.Module ] ] = None,
    func_intensity : Optional[ Type[ nn.Module ] ] = None 
    ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray ]:
    """
    gan_preprocessing:
    將 dicom 檔轉換成 GAN 的輸入

    Args:
    ----------
    dicom: Union[ DicomDir, str ]
    要讀取的檔案或位置
    ----------
    pkl: str
    要讀取的 pickle 檔案位置

    Return:
    ---------
    Tuple[ np.ndarray, np.ndarray, np.ndarray ]
    依序為 ct, air, bone
    """
    denoise_bound = ( -500, -499 )
    view_bound = ( -500, 500 )

    if isinstance( dicom, str ) and pkl is None:
        dicom = dcmread( dicom )
        
    elif isinstance( pkl, int ):
        dicoms = path_prepare_0915( path_to_dataset = path_to_dataset )
        dicoms.sort()
        dicom = dcmread( dicoms[ pkl ] )
    
    img = dicom.pixel_array.copy()

    img_hu = transform_to_hu( dicom, img )
    
    mask = DenoiseMask( bound = denoise_bound, always_apply = True )( image = img_hu )[ "image" ]
    
    img = img * mask

    img = transform_to_hu( dicom, img )
    img = hu_clip( img, view_bound, None, False )

    air_bound = ( -301, -300 )
    bone_bound = ( 200, 201 )

    sample = AirBoneMask( 
            bound = view_bound,
            air_bound = air_bound,
            bone_bound = bone_bound,
            always_apply = True,
    )( image = img )[ "image" ]

    air, bone = sample[ 0, : , : ], sample[ 1, : , : ]

    bone = refine_mask( bone, bone )

    img = np.expand_dims( img, 0).astype(np.float32)
    air = np.expand_dims(air, 0).astype(np.float32)
    bone = np.expand_dims(bone, 0).astype(np.float32)

    return img, air, bone
    
    