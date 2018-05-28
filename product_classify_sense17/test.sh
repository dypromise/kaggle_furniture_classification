srun -p Test -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=9 \
python -u /mnt/lustre17/yangkunlin/fur_dy/code/imaterialist-furniture-2018/main.py /mnt/lustre17/yangkunlin/fur_dy/data \
--mode=test \
--test-prob=True \
--model-name=se_resnext101_32x4d \
--input-size=224 \
--add-size=32 \
--batch-size=32 \
--checkpoint-file=/mnt/lustre17/yangkunlin/fur_dy/y2d/seResnext101/checkpoint1_best.pth \
--test-predsfile=/mnt/lustre17/yangkunlin/fur_dy/data/npys/se_resnext101_32x4d_pred.npy

