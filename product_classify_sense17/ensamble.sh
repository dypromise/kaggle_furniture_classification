srun -p Test -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=9 \
python -u /mnt/lustre17/yangkunlin/fur_dy/code/imaterialist-furniture-2018/ensamble.py