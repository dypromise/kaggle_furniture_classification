# srun -p VIBackEnd2 -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=6 \
python3 -u ./imaterialist-furniture-2018/main.py /media/bigdrive/dingyang/data \
--mode=train \
--model-name=resnet152 \
--input-size=224 \
--add-size=32 \
--batch-size=16 \
--checkpoint-file=./checkpoint_files/res152_weightedloss.pth \
--epochs=40 \
> log_res152_wl.txt 2>&1 
