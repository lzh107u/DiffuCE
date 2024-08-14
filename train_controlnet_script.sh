accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
 --output_dir="controlnet-training-default-output" \
 --train_data_dir="ct-dataset-0905.py" \
 --conditioning_image_column=cond \
 --image_column=image \
 --caption_column=prompt \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "dataset/CBCT_img_folder_v2/000/folder_0_id_0_name_CT.1.2.246.352.62.1.4626535837401668255.17432788409501768577.dcm.png" \
 --validation_prompt "a clean CT image" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --use_8bit_adam \
 --num_train_epochs=3 \
 --tracker_project_name="controlnet" \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \
 --max_train_steps=10 \