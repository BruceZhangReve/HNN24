#python train.py --task gc --dataset IMDBMULTI --model HKPNet  --normalize-feats 1 --log-freq 5   --epochs 5000  --patience 1000  --lr 0.005  --dim 16 --num-layers 6 --batch-size 32 --val-prop 0.1 --test-prop 0 --cuda 0 --kernel-size 7

#BKN
python train.py \
    --lr 2e-4 --dropout 0 --cuda 3 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 1234 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset IMDBMULTI --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1234 --split_graph False

python train.py \
    --lr 2e-4 --dropout 0 --cuda 3 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 1234 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 3 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset IMDBMULTI --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1234 --split_graph False

python train.py \
    --lr 2e-4 --dropout 0 --cuda 3 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 1234 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 5 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset IMDBMULTI --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1234 --split_graph False

python train.py \
    --lr 2e-4 --dropout 0 --cuda 3 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 1234 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset IMDBMULTI --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1234 --split_graph False

python train.py \
    --lr 2e-4 --dropout 0 --cuda 3 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 1234 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 7 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset IMDBMULTI --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1234 --split_graph False

python train.py \
    --lr 2e-4 --dropout 0 --cuda 3 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 1234 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 8 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset IMDBMULTI --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1234 --split_graph False

python train.py \
    --lr 2e-4 --dropout 0 --cuda 3 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 1234 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 9 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset IMDBMULTI --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1234 --split_graph False
