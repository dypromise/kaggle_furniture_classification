srun -p VIBackEnd2 -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=6 \
python -u /mnt/lustre17/yangkunlin/fur_dy/code/imaterialist-furniture-2018/main.py /mnt/lustre17/yangkunlin/fur_dy/data \
--mode=train \
--model-name=inceptionv4 \
--input-size=299 \
--add-size=42 \
--batch-size=16 \
--checkpoint-file=/mnt/lustre17/yangkunlin/fur_dy/data/checkpoint_files/incepv4_weightedloss.pth \
--epochs=40 \
> log_incepv4_wl.txt 2>&1 
