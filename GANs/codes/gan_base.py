from typing import Any, Union
import torch 
import torch.nn as nn
import numpy as np
from pydicom.dicomdir import DicomDir
from PIL import Image

from .dataset import gan_preprocessing

class GANBaseClass :
    """
    GANBaseClass:
    作為各 GAN network 的基底，定義了 __call__ 的具體操作。
    """
    def __init__( 
        self, 
        data_mode : str = 'dicom',
        device : torch.device = 'cuda',
        dtype : torch.dtype = torch.float32 
        ) -> None:
        # Initialize the fundamental parameters.
        self.network : Union[ nn.Module, None ] = None
        self.device : torch.device = device
        self.dtype : torch.dtype = dtype
        self.data_mode = data_mode

    def set_env( 
        self, 
        device : torch.device = 'cuda',
        dtype : torch.dtype = torch.float32 
        ) -> None:

        self.device = device
        self.dtype = dtype
        
    def to_pil( 
        self,
        img : np.ndarray 
    ) -> Image.Image:
        img = img - np.min( img )
        img = img / np.max( img )

        img *= 255
        img = np.uint8( img )
        img = Image.fromarray( img, 'L' ).resize( ( 512, 512 ) )

        return img

    def __call__( self, x : Union[ torch.Tensor, DicomDir ] ) -> Union[ torch.Tensor, DicomDir, Image.Image ]:
        """
        __call__:
        執行一次 artifact removal 運算

        Args:
        ----------
        x : Union[ torch.Tensor, DicomDir ]
        輸入影像

        Return:
        ----------
        np.ndarray

        """
        if isinstance( x, DicomDir ) or ( self.data_mode == 'dicom' and isinstance( x, str ) ):
            # 若輸入為 dicom 檔，則透過 gan_preprocessing 取得 x 
            # 或是輸入為字串，且檔案模式指定為 `dicom`
            images = gan_preprocessing( dicom = x )
            x = images[ 0 ]
            x = torch.from_numpy( x )
        
        if not isinstance( self.network, nn.Module ):
            # 若不存在 self.network ，則離開
            print( "GANBaseClass, Warning: No network is provided." )
            return x
        
        # 確保 x 的維度有 4 維
        ## [ 1, c, H, W ]
        while len( x.shape ) < 4:
            x = torch.unsqueeze( input = x, dim = 0 )

        # 載入 GPU
        self.network.eval()
        self.network.to( device = self.device, dtype = self.dtype )
        x = x.to( device = self.device, dtype = self.dtype )
        
        # 計算
        x = self.network( x )

        # 釋出 GPU
        x = x.cpu().detach().numpy()
        self.network.to( device = 'cpu' )

        x = self.to_pil( img = np.squeeze( x ) )

        return x