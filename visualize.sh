python3 visualize_attention.py \
    --arch vit_base \
    --patch_size 8 \
    --pretrained_weights ../pretrained/dino_vitbase8_pretrain_full_checkpoint.pth \
    --image_path $1 \
    --output_dir $2 \
    --threshold $3 