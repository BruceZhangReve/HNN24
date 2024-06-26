#python train.py --task nc --dataset squirrel --model HKPNet --dim 16 --lr 0.005 --num-layers 3 --act relu --bias 1 --dropout 0.2 --weight-decay 1e-4 --manifold Lorentz --log-freq 5 --patience 1500 --linear-before 16 --cuda 0 --kernel-size 4
#Run via HKN
#python train.py --task nc --dataset film --model HKPNet --dim 16 --lr 0.005 --num_layers 2 --act relu --bias 1 --dropout 0.05 --weight_decay 1e-4 --manifold Lorentz --log_freq 5 --patience 50 --linear_before 16 --cuda 0 --kernel_size 8 --epochs 500 --use_geoopt True --min_epochs 100
#Run via BKN
python train.py --task nc --dataset squirrel --model BKNet --dim 16 --lr 0.001 --num_layers 2 --act relu --bias 1 --dropout 0.25 --weight_decay 1e-3 --manifold PoincareBall --log_freq 5 --patience 50 --linear_before 64 --cuda 0 --kernel_size 8 --use_geoopt False --c 1.0 --act relu --epochs 500 --min_epochs 100 --seed 8 --batch_size 32 --cuda -1