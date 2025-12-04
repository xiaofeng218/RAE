hf download --repo-type dataset ILSVRC/imagenet-1k --local-dir /data/dataset/imagenet-1k --include README.md classes.py
python preprocess_imagenet_data.py
hf download nyu-visionx/RAE-collections --local-dir models --include discs/dino_vit_small_patch8_224.pth

torchrun --standalone \
  src/train_stage1.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --data-path /data/dataset/imagenet-1k/one-sample-train \
  --results-dir results/stage1 \
  --image-size 256 --precision fp16 --one-sample
#   --ckpt <optional_ckpt> \

