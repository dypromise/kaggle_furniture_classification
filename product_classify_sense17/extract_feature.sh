srun -p VIBackEnd2 -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=9 \
python -u /mnt/lustre17/yangkunlin/fur_dy/code/imaterialist-furniture-2018/feature_extractor.py /mnt/lustre17/yangkunlin/fur_dy/data \
--model-name=inceptionresnetv2 \
--input-size=299 \
--add-size=42 \
--batch-size=64 \
--checkpoint-file=/mnt/lustre17/yangkunlin/fur_dy/ykl_to_dy/inceptionResnetV2/checkpoint1_best.pth \
--feature-file=/mnt/lustre17/yangkunlin/fur_dy/ykl_to_dy/inceptionResnetV2/fts_incepresv2.npy
