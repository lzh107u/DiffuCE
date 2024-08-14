export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="training-lora-default-output"
export HUB_MODEL_ID="your-huggingface-repo-id"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export INSTANCE_DIR="dataset/CHEST/*/CT/CT*.dcm"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --validation_prompt="a clean CT image" \
  --instance_prompt="a clean CT image" \
  --use_8bit_adam \
  --pre_compute_text_embeddings \
