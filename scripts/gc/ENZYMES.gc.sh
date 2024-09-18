#python train.py --task gc --dataset ENZYMES --model HKPNet  --normalize-feats 1 --log-freq 5   --epochs 5000  --patience 1000  --lr 0.05  --dim 16 --dropout 0.05 --num-layers 4 --batch-size 32 --val-prop 0.1 --test-prop 0 --cuda 0 --kernel-size 4

#GCN
#python train.py \
    #--lr 0.002 --dropout 0 --cuda 2 --epochs 10000 --weight_decay 0 --optimizer adam --momentum 0.999 --patience 500 --seed 10 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt False --AggKlein False --corr 1 --nei_agg 0 --task gc --model GCN --dim 64 --manifold Euclidean --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    #--dataset ENZYMES --batch_size 600 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 10 --split_graph False

#HGCN
#python train.py \
    #--lr 1e-3 --dropout 0 --cuda 2 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 50 --seed 10 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt False --AggKlein False --corr 1 --nei_agg 0 --task gc --model HGCN --dim 64 --manifold Hyperboloid --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    #--dataset ENZYMES --batch_size 600 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 10 --split_graph False

#HKPNet
#python train.py \
    #--lr 0.005 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 100 --seed 42 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 250 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task gc --model HKPNet --dim 64 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    #--dataset ENZYMES --batch_size 100 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 42 --split_graph False

#BKN
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 1 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 5 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 1 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 1 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 8 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 1 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 9 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 1 --split_graph False


python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 8 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 8 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 3 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 8 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 5 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 8 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 8 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 7 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 8 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 8 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 8 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 9 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False


python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 24 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 24 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 24 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 3 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 24 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 24 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 5 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 24 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 24 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 24 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 24 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 7 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 24 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 24 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 8 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 24 --split_graph False
python train.py \
    --lr 2e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 120 --seed 24 --log_freq 5 --eval_freq 5 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task gc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 9 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset ENZYMES --batch_size 32 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 24 --split_graph False