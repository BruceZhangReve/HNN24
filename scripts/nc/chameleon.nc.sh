#python train.py --task nc --dataset chameleon --model HKPNet --dim 16 --lr 0.005 --num-layers 3 --act relu --bias 1 --dropout 0.2 --weight-decay 1e-4 --manifold Lorentz --log-freq 5 --patience 1500 --linear-before 16 --cuda 0 --kernel-size 6

#HKN_neoe
#python train.py \
    #--lr 1e-3 --dropout 0.4 --cuda 7 --epochs 5000 --weight_decay 1e-3 --optimizer radam --momentum 0.999 --patience 50 --seed 8 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 250 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 64 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    #--dataset chameleon --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False

#BKNet
python train.py \
    --lr 1e-3 --dropout 0.02 --cuda 7 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 50 --seed 25 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 150 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 2 --task nc --model BKNet --dim 52 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 52 \
    --dataset chameleon --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 25 --split_graph False