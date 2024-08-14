from autoencoder_kl_v2 import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from typing import List
from diffusers.models.attention_processor import Attention, LoRAAttnProcessor

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'
SAMPLE_NAME = '17795.jpg'
resolution = 512

image_transforms = transforms.Compose(
        [
            transforms.Resize( resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop( resolution ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def go_through_pipeline( 
    model: AutoencoderKL = None,
    sample_name: str = SAMPLE_NAME,
    ) -> None:
    
    image = Image.open( sample_name ).resize( ( 512, 512 ) ).convert( "RGB" )
    image_processor = VaeImageProcessor( vae_scale_factor = 8 )
    
    input: torch.Tensor = image_transforms( image )
    input = torch.unsqueeze( input, dim = 0 )

    model.to( device = device )
    model.eval()
    input = input.to( device = device )

    embedding = model.encode( input.to( device = device ) ).latent_dist.sample()
    embedding = model.config.scaling_factor * embedding

    image : torch.Tensor = model.decode( embedding / model.config.scaling_factor, return_dict = False )[ 0 ]
    image = image.cpu().detach()
    result : List[ Image.Image ] = image_processor.postprocess( image = image )
    
    result[ 0 ].save( 'result_lora.jpg' )

    return 

def train( model : AutoencoderKL ):
    # model.encoder.set_gradient_flag()
    unit = model.encoder.mid_block.attentions[ 0 ].processor
    print( 'type of processor: {}'.format( type( unit ) ) )
    state_dict = unit.state_dict()
    for name in state_dict:
        print( name )

def main():
    pretrained_vae = AutoencoderKL.from_pretrained( pretrained_model_name_or_path = pretrained_model_name_or_path, subfolder = 'vae', low_cpu_mem_usage = False )
    pretrained_vae.to( device = device )
    # go_through_pipeline( model = pretrained_vae )
    train( model = pretrained_vae )
    return 

def unit_test():
    attn_proc = Attention( query_dim = 64, processor = LoRAAttnProcessor( hidden_size = 64 ) )
    attn_proc.to( device = device )

if __name__ == '__main__':
    main()
    # unit_test()