from PIL import Image
from diffusers import AutoencoderKL
import torch



def encode_latent(
    image : Image.Image,
    batch_size : int = 1,
    ) -> torch.Tensor:

    
    return 