#Run via HKN
#python train.py --task nc --dataset cornell --model HKPNet --dim 16 --lr 1e-3 --num_layers 2 --act relu --bias 1 --dropout 0.5 --weight_decay 1e-3 --manifold Lorentz --log_freq 5 --patience 50 --linear_before 32 --cuda 0 --kernel_size 6 --epochs 500 --use_geoopt True --seed 8 --KP_extent 0.8 --optimizer radam --margin 2.0
#Run via BKN
#python train.py --task nc --dataset cornell --model BKNet --dim 16 --lr 1e-3 --num_layers 2 --act relu --bias 1 --dropout 0.5 --weight_decay 1e-3 --manifold PoincareBall --log_freq 5 --patience 250 --linear_before 32 --cuda 0 --kernel_size 6 --use_geoopt False --c 1.0 --act relu --epochs 500 --min_epochs 200 --seed 8 --optimizer radam --log_freq 1 --KP_extent 0.66 --margin 2.0
python train.py \
    --lr 5e-4 --dropout 0.2 --cuda 7 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 50 --seed 123 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 300 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 2 --task nc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset cornell --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 123 --split_graph False
