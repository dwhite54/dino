python -m torch.distributed.launch --nproc_per_node=4 eval_knn.py \
	--data_path_train /home/ubuntu/afex-1k/train/ \
	--data_path_test /home/ubuntu/afex-1k/test/ \
	--num_workers 16 \
	--pretrained_weights pretrained/dino_vitbase8_pretrain_full_checkpoint.pth \
	--arch vit_base \
	--patch_size 8 \
    --nb_knn 1 \
    --dump_features /home/ubuntu/afex-1k/knn-eval/
