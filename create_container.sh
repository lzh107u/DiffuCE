# Create the Image: DiffuCE_Image
# - Build the image
# - Move datasets and model weights to correct places
# Decompress the model weights and datasets

export IMAGE_NAME="diffuce_image_v1"
export CONTAINER_NAME="diffuce_v1"

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
docker build -f dockerfile_DiffuCE -t $IMAGE_NAME .

# Create teh Container: diffuce_image_v1
docker run -it -d --gpus all --name $CONTAINER_NAME -v ./:/workspace/DiffuCE -p 8878:8888 --shm-size=16g $IMAGE_NAME
