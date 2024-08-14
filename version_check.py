import diffusers
import scipy
import pydicom
import pywt
import segmentation_models_pytorch
import cv2
import albumentations 
import nibabel
def show( name, version ):
    print("{}: {}".format( name, version ) )

print( "diffusers: {}".format( diffusers.__version__ ) )
show( "scipy", scipy.__version__ )
show( "pydicom", pydicom.__version__ )
show( "pywt", pywt.__version__ )
show( "segmentation models pytorch", segmentation_models_pytorch.__version__)
show( "cv2", cv2.__version__ )
show( "albumentation", albumentations.__version__ )
show( "nibabel", nibabel.__version__ )