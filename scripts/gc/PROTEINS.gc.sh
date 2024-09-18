#python train.py --task gc --dataset PROTEINS --model HKPNet  --normalize-feats 1 --log-freq 5   --epochs 5000  --patience 1000  --lr 0.005  --dim 16 --num-layers 6 --batch-size 32 --val-prop 0.1 --test-prop 0 --cuda 0 --kernel-size 4

#GCN
#python train.py \
    #--lr 0.005 --dropout 0 --cuda 2 --epochs 5000 --weight_decay 0 --optimizer adam --momentum 0.999 --patience 50 --seed 12345 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 50 \
    #--use_geoopt False --AggKlein False --corr 0 --nei_agg 0 --task gc --model GCN --dim 16 --manifold Euclidean --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 6 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    #--dataset PROTEINS --batch_size 256 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 12345 --split_graph False

#HKPNet
#python train.py \
    #--lr 0.005 --dropout 0 --cuda 0 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 50 --seed 1234 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task gc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    #--dataset PROTEINS --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1234 --split_graph False

#BKN
python train.py \
    --lr 2e-3 --dropout 0 --cuda 0 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 1234 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 50 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset PROTEINS --batch_size 128 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1234 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0 --cuda 0 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 1234 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 50 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 7 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset PROTEINS --batch_size 128 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1234 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0 --cuda 0 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 1234 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 50 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 8 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset PROTEINS --batch_size 128 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1234 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0 --cuda 0 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 1234 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 50 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 9 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset PROTEINS --batch_size 128 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1234 --split_graph False


python train.py \
    --lr 2e-3 --dropout 0 --cuda 0 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 12345 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 50 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset PROTEINS --batch_size 128 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 12345 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0 --cuda 0 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 12345 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 50 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 3 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset PROTEINS --batch_size 128 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 12345 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0 --cuda 0 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 12345 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 50 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 5 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset PROTEINS --batch_size 128 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 12345 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0 --cuda 0 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 12345 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 50 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset PROTEINS --batch_size 128 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 12345 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0 --cuda 0 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 12345 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 50 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 7 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset PROTEINS --batch_size 128 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 12345 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0 --cuda 0 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 12345 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 50 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 8 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset PROTEINS --batch_size 128 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 12345 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0 --cuda 0 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 120 --seed 12345 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 50 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 16 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 9 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset PROTEINS --batch_size 128 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 12345 --split_graph False