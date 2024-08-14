from typing import Any, List, Optional
from diffusers import (
    ControlNetModel,
    UniPCMultistepScheduler
)
from custom_backward import (
    StableDiffusionModifiedControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgModifiedPipeline,
)
import torch
from peft_lora_pipeline import merging_lora_with_base
from PIL import Image

DEFAULT_CHECKPOINTS = {
    'air' : 'ControlNet-022-0912-air-la',
    'bone' : 'ControlNet-025-0922-bone-lung-la',
    'wavelet' : 'ControlNet-024-0921-wavelet-lung-la',
    'tissue' : 'ControlNet-026-1225-tissue-hu',}
SYNTHRAD_CHECKPOINTS = {
    'air' : 'synthrad2023_air',
    'bone' : 'synthrad2023_bone',
    'wavelet' : 'synthrad2023_wavelet',
}

class CDDM :
    def __init__( 
        self,
        mode_names : List[ str ] = [ 'air', 'bone', 'wavelet' ],
        checkpoint : Optional[ int ] = None,
        cdd_state : Optional[ str ] = 'normal',
        dataset_mode : Optional[ str ] = 'default',
        pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5',
        lora_ckpt_dir : Optional[ str ] = 'CDD_lora',
        ) -> None:
        """
        Conditional Denoising Diffusion Models

        用於對 CT-like noisy latent 進行 denoise。
        
        Args:
        ----------
        mode_names: List[ str ]
        ControlNet 的順序，注意在實際使用時也需要按照這裡的順序
        ----------
        checkpoint : Optional[ int ]
        同一次訓練，不同 checkpoint 的版本，預設為 None，即最後一組
        ----------
        dataset_mode: Optional[ str ]
        資料集模式，選取不同的權重以進行最佳推論。目前支援以下模式：
        - `default`
        - `synthrad2023`

        """
        
        # 建立 pipeline
        if dataset_mode not in [ 'default', 'synthrad2023' ]:
            print( 'CDD, init: Invalid `dataset_mode` is given: {}, set to `default`.'.format( dataset_mode ) )
            dataset_mode = 'default' 

        if dataset_mode == 'default':
            self.checkpoint_folders = DEFAULT_CHECKPOINTS
        elif dataset_mode == 'synthrad2023':
            self.checkpoint_folders = SYNTHRAD_CHECKPOINTS
        
        self.pretrained_folder_prefix = 'pretrained_weights/ControlNet/'
        self.pipe = self._prepare_pipeline( 
            mode = mode_names, 
            checkpoint = checkpoint, 
            cdd_state = cdd_state,
            pretrained_model_name_or_path = pretrained_model_name_or_path,
            lora_ckpt_dir = lora_ckpt_dir )


    
    def _prepare_pipeline( 
        self, 
        mode : List[ str ], 
        checkpoint : Optional[ int ] = None,
        cdd_state : Optional[ str ] = 'normal', 
        pretrained_model_name_or_path : Optional[ str ] = 'runwayml/stable-diffusion-v1-5',
        lora_ckpt_dir : Optional[ str ] = 'CDD_lora',
        ) -> StableDiffusionModifiedControlNetImg2ImgPipeline:
        """
        prepare_pipeline:
        根據不同的 condition 載入不同的 controlnet 權重

        Args:
        ----------
        mode: List[ str ]
        controlnet 模式，有以下幾種選擇：
            air: air-mask
            bone: bone-mask
            wavelet: Level-2 wavelet transformation low frequency component
        ----------
        checkpoint: Optional[ int ] = None
        指定載入某個 controlnet 模式下的特定 checkpoint
        ----------
        cdd_state: Optional[ str ]
        決定 CDD 內 U-Net pipeline 的類型，目前支援以下數種模式：
        1. normal: DiffuCE 正常模式，帶有 `ControlNet` 與 `LoRA` 
        2. ablation: Ablation 模式，不帶有 `ControlNet`，僅含 `LoRA`
        ----------
        pretrained_model_name_or_path: Optional[ str ], default: `runwayml/stable-diffusion-v1-5`
        決定 CDDM 的基底模型權重類別，這裡是 HuggingFace Diffusers 的 API
    
        Return:
        ----------
        pipe: StableDiffusionControlNetImg2ImgPipeline
        載入權重後的 controlnet pipeline

        """
        def set_single_controlnet( 
            name: str,
            pretrained_model_name_or_path : Optional[ str ] = 'runwayml/stable-diffusion-v1-5' 
            ) -> ControlNetModel:
            try:
                pretrained_folder = self.checkpoint_folders[ name ]
            except ValueError:
                raise ValueError( 'Invalid mode name: {}'.format( name ) )
    

            pretrained_folder = self.pretrained_folder_prefix + pretrained_folder
            if checkpoint is not None:
                pretrained_folder = pretrained_folder + '/checkpoint-{}'.format( checkpoint )
        
            print( "CDDM loads controlnet model: {}".format( pretrained_folder ) )
    
            return ControlNetModel.from_pretrained( pretrained_folder )
        
        # 檢查 `cdd_state` 是否合規
        if cdd_state not in [ 'normal', 'ablation' ]:
            print( 'CDD._prepare_pipeline: invalid `cdd_state`: {}'.format( cdd_state ) )
            cdd_state = 'normal'

        # 依據 `cdd_state` 宣告 diffusion pipeline
        if cdd_state == 'normal':
            controlnet = [ set_single_controlnet( name = name ) for name in mode ]
            pipe : StableDiffusionModifiedControlNetImg2ImgPipeline = StableDiffusionModifiedControlNetImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path, 
                controlnet = controlnet, 
                torch_dtype = torch.float32 
            )
        elif cdd_state == 'ablation':
            pipe : StableDiffusionImg2ImgModifiedPipeline = StableDiffusionImg2ImgModifiedPipeline.from_pretrained(
                pretrained_model_name_or_path = pretrained_model_name_or_path,
                torch_dtype = torch.float32
            )

        pipe.scheduler = UniPCMultistepScheduler.from_config( pipe.scheduler.config )
        pipe = merging_lora_with_base( pipe = pipe, ckpt_dir = lora_ckpt_dir, adapter_name = 'prior2ct' ) # peft convention
        
        return pipe
    

    def denoise( 
        self, 
        image : Image.Image,
        cond : List[ Image.Image ],
        latent : torch.Tensor,
        device : torch.device = 'cuda',
        prompt : str = 'a clean CT image',
        num_inference_steps : int = 40,
        strength : float = 0.2,
        scales : List[ float ] = [ 0.1, 0.05, 0.2 ],
        random_seed : Optional[ int ] = 0,
        ) -> torch.Tensor:
        """
        CDDM.denoise:
        實際對一組 CT-like noisy latent 與相對應的 conditions 進行降噪

        Args:
        ----------
        image : Image.Image
        原圖，可忽略
        ----------
        cond : List[ Image.Image ]
        用於各 ControlNet 的條件輸入，注意這裡的排列順序要與宣告時使用的 mode_names 相同
        ----------
        latent : torch.Tensor
        CT-like noisy latent，降噪目標
        ----------
        device : torch.device
        運行位置，預設為 `cuda`
        ----------
        prompt : str
        使用 CDDM 時搭配的 text prompt ，預設為 `a clean CT image`
        ----------
        num_inference_steps : int
        降噪總步數，設置愈小，跨步愈大。
        這裡預設為 40 步
        ----------
        strength : float
        噪聲強度，範圍介於 0 到 1，數值愈大，噪聲強度愈強。
        注意：
            strength 太小時，CBCT 中的 Streak Artifact 無法去除
            strength 太大時，latent 中的細節資訊被過度抹除
        這裡預設為 0.2 ，即到 200 步的位置。
        ----------
        scales : List[ float ]
        各 ControlNet 的強度，愈設為：
            air : 0.1
            bone : 0.05
            wavelet : 0.2
        ----------
        random_seed : Optional[ int ]
        隨機種子，愈設為 0


        Return:
        ----------
        torch.Tensor 
        CDDM 降噪後的 clean CT-like latent ，
        準備交付 CRD 進行下游重建任務
        
        """
        # gpu load
        self.pipe.to( device )

        generator = torch.Generator( device = device ).manual_seed( random_seed )

        if isinstance( self.pipe, StableDiffusionModifiedControlNetImg2ImgPipeline ):
            output = self.pipe(
                prompt = prompt,
                image = image,
                control_image = cond,
                num_inference_steps = num_inference_steps,
                generator = generator,
                controlnet_conditioning_scale = scales,
                modified_latent = latent,
                strength = strength,
                early_stop_step = 0,
                output_type = "latent",
            )
        elif isinstance( self.pipe, StableDiffusionImg2ImgModifiedPipeline ):
            output = self.pipe(
                prompt = prompt,
                image = image,
                num_inference_steps = num_inference_steps,
                generator = generator,
                modified_latent = latent,
                strength = strength,
                early_stop_step = 0,
                output_type = "latent"
            )

        # gpu offload
        self.pipe.to( 'cpu' )

        return output