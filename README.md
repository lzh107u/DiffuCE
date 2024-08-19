# DiffuCE: Expert-Level CBCT Image Enhancement using a Novel Conditional Denoising Diffusion Model with Latent Alignment

This is the official repository of the paper DiffuCE, aiming for CBCT image enhancement via latent diffusion model framework.

## Installation

This repository requires Huggingface Diffusers and PEFT library, and these actions are included in the bash scripts.

To install this repository, please follow these steps:
```
git clone https://github.com/lzh107u/DiffuCE.git
```

The pretrained weights and datasets can be found in IIR Google Drive with the following path:
```
iir_file/111(2021)碩_XX_XX_X泓_XX/畢業光碟/X泓/DiffuCE/docker_weights.zip
```
Due to the privacy of the patients, the files are not available to public. Please contact us if you need to access these files.

Please put `docker_weights.zip` under the folder `DiffuCE` like the following structure:
```
DiffuCE/
  docker_weights.zip   <--
  create_image.sh
  create_container.sh
  custom_backward/
  vae_lora_test/
  ...
  
```
Make sure the `docker_weights.zip` is at the right place to prevent from incorrect installation.

## Build Container/Image

You can modify the `IMAGE_NAME` and `CONTAINER_NAME` in the file `create_container.sh`.

To build the container, please follow the instruction:
```
bash create_container.sh
```

If you only want to build the image instead of starting a container, please follow the instruction:
```
bash create_image.sh
```

## Model Weights

The DiffuCE framework can be divided into three parts: DBE(Encoder), CDD(Denoiser), and CRD(Decoder).

### DBE - Domain Bridging Encoder

The main function of DBE is defined in the file `vae_lora_test/backward_experiment.py`, and the default model weight is `vae_lora_test/lora_unit_epoch1000_mixed-ts.pt`.

If you have trained new weight for DBE, please make sure the weight is under the folder `vae_lora_test` and modify the weight list in the `backward_experiment.py`.

```
DiffuCE/
  vae_lora_test/          <-- Put your new weight under this folder
    custom_backward/
    __init__.py
    lora_unit_epoch1000_mixed-ts.pt   <-- Default DBE weight
    backward_experiment.py            <-- DBE main function
    ...    
```

### CDD - Conditional Diffusion Denoiser

The weight of CDD is based on pre-trained Stable Diffusion, and the fine-tuning involves `ControlNet` and `LoRA`.

The weight of LoRA follows the convention of Huggingface PEFT library, and the default weight can be found in `docker_weights.zip`.
This folder is named `CDD_lora` and moved to default position via the `create_container.sh`.

The weights of ControlNet follow the convention of Huggingface Diffusers library, and the default weights can also be found in `docker_weights.zip`.
The folder is named `ControlNet` and moved to default position via the `create_container.sh`

Note: 
- Huggingface Diffusers library only support `.safetensors` format in the recent versions, so please make sure to transform the old model weights with format `.bin` to the new format. 
- Modify the paths defined in the `cddm.py` if you add new weight.
```
DiffuCE/
  CDD_lora/    <-- Default layout of LoRA fine-tuning via PEFT library
    logs/
    text_encoder/
    unet/
  pretrained_weights/
    ControlNet/
      ControlNet-022-0912-air-la/
        config.json
        diffusion_pytorch_model.safetensors    <-- Make sure the model weight is `.safetensors`
      ControlNet-024-0921-wavelet-lung-la/
      ...
    decoder/
    ...
  DiffuCE.py   <-- Main function of DiffuCE framework
  cddm.py      <-- Main function of CDD
  ...
```

### CRD - Conditional Refinement Decoder

The main function of CRD is defined in `cond_refine_decoder_utils.py`, and the default model weight is `pretrained_weights/decoder/decoder-009/decoder+lora_final.pkl`.
The folder `decoder` can be found in `docker_weights.zip`, and it will be moved to its position via `create_container.sh`

```
DiffuCE/
  pretrained_weights/
    ControlNet/
    decoder/
      decoder-009/
        decoder+lora_final.pkl
        ...
  DiffuCE.py    <-- Main function of DiffuCE framework
  cond_refine_utils.py    <-- Main function of CRD
```

## Run Scripts

Note:
- The training/inference procedure includes some functions from Huggingface Diffusers, which might take you few seconds or minutes to cache some model weights in the first execution.

### Data Preprocessing

The preprocessing of input CT images and its corresponding conditions is defined in `dicom_utils.py`.

To obtain a set of demo images(a processed CT image and its conditions), please follow the instruction:
```
python dicom_utils.py
```

The preprocessing of pseudo-CBCT is defined in `sino_utils.py`.

To run the demo script of generating the pair pseudo-CBCT image, please follow the instruction:
```
python sino_utils.py
```

### DiffuCE Inference

To run the DiffuCE framework, please follow the instruction:
```
python DiffuCE.py
```

To modify the arguments, such as dicom filename and coefficients of control modules, please check the function `inference` or dive into other dependent files.

### Train DBE

To run the training script of DBE, please follow the instruction:
```
python tune_alignment.py
```
The default output is as follows:
```
DiffuCE/
  DBE_tuning/
    loss_history.png    <-- Loss curve of training
  alignment_unit_default_name.pt  <-- Model weight
  tune_alignment.py
  ...
```

### Train CDD

The training of CDD is divided into two parts: LoRA and ControlNet.

To train the ControlNet, please follow the instruction:
```
bash train_controlnet_script.sh
```
**ATTENTION**: The dataset for this training is customized, which requires your manual check before the training starts. There will be a message popping out and ask you to choose `y/N` for generating this custom dataset.

To train the LoRA, please follow the instruction:
```
bash train_lora_script.sh
```

The default output is as follows:
```
DiffuCE/
  controlnet-training-default-output/  <-- Default ouput of ControlNet training
    config.json
    diffusion_pytorch_model.safetensors
    logs/
  training-lora-default-output/        <-- Default output of LoRA training
    pytorch_lora_weights.safetensors
    logs/
  train_controlnet_script.sh    <-- Run this script to train ControlNet
  train_controlnet.py      
  train_lora_script.sh          <-- Run this script to train LoRA
  train_dreambooth_lora.py
  ct-dataset-0905.py      <-- Custom dataset 
  ...
```

Note:
- In the training of LoRA, the `text encoder` is fixed. If you want to modify the setting, please check the official site of Huggingface for more details.

### Train CRD

To run the training script of CRD, please follow the instruction:
```
python tune_decoder.py
```

The default output is as follows:
```
DiffuCE/
  decoder+lora-best-XXXX.pkl     <-- model weights
  decoder+lora_final_XXXX.pkl    <-- XXXX stands for timestamp
  loss_history_decoder_XXXX.png
  loss_history_decoder_sub_XXXX.png
  tune_decoder.py    <-- Train decoder
  ...
```
