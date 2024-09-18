#HKN
#python train.py --task nc --dataset cora --model HKPNet --dim 16 --lr 0.0005 --num-layers 2 --act relu --bias 1 --dropout 0.9 --weight-decay 1e-4 --manifold Lorentz --log-freq 5 --patience 1500 --linear-before 16 --grad-clip 0.5 --cuda 0 --kernel-size 2

#HGCN
#python train.py \
    #--lr 1e-3 --dropout 0.5 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 50 --seed 25 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 250 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt False --AggKlein False --corr 0 --nei_agg 0 --task nc --model HGCN --dim 16 --manifold Hyperboloid --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    #--dataset cora --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 25 --split_graph False

#HKN_neoe fine setting
#python train.py \
    #--lr 1e-3 --dropout 0.75 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 5 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    #--use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 36 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 36 \
    #--dataset cora --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 5 --split_graph False

#this seem to be better
python train.py \
    --lr 1e-3 --dropout 0.7 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 5 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 48 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 48 \
    --dataset cora --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 5 --split_graph False

#BKNet
#python train.py \
    #--lr 2e-3 --dropout 0.7 --cuda 6 --epochs 5000 --weight_decay 1e-3 --optimizer radam --momentum 0.999 --patience 50 --seed 5 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 150 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task nc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 3 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 3 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    #--dataset cora --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 5 --split_graph False