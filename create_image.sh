export CONTAINER_NAME="diffuce_image_v1"

# Decompress the model weights and datasets
unzip docker_weights.zip

mv -n docker_weights/0915_tuning ./
mv -n docker_weights/CBCT_img_folder_v2 ./dataset/
mv -n docker_weights/CHEST ./dataset/
mv -n docker_weights/ControlNet/* ./pretrained_weights/ControlNet/
mv -n docker_weights/decoder/* ./pretrained_weights/decoder/
mv -n docker_weights/CDD_lora ./
mv -n docker_weights/lora_unit_epoch1000_mixed-ts.pt ./vae_lora_test/
# rm -r docker_weights

# Build Image
docker build -f dockerfile_DiffuCE -t $CONTAINER_NAME .

