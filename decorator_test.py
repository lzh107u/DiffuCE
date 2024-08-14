from training_utils import write_file
from dicom_utils import dicom_pipeline
from glob import glob

@write_file
def test( file_path : str ):

    ds_paths = glob( 'dataset/CHEST/*/CT/CT*.dcm' )
    rets = dicom_pipeline( ds_dir = ds_paths[ 10 ] )
    
    rets[ 0 ].save( file_path )
    return

if __name__ == "__main__":

    test( file_path = 'test_folder/test_image.png' ) 