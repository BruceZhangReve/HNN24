#BKN corr=0,agg=2
#cornell
#python train.py \
    #--lr 1e-3 --dropout 0.2 --cuda 7 --epochs 5000 --weight_decay 5e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 7 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 32 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 7 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
    #--dataset cornell --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 7 --split_graph False
#python train.py \
    #--lr 1e-3 --dropout 0.2 --cuda 7 --epochs 5000 --weight_decay 5e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 42 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 32 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 7 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
    #--dataset cornell --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 28 --split_graph False
#python train.py \
    #--lr 1e-3 --dropout 0.2 --cuda 7 --epochs 5000 --weight_decay 5e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 42 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 32 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 7 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
    #--dataset cornell --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 28 --split_graph False
#texas
#python train.py \
    #--lr 1e-3 --dropout 0.2 --cuda 7 --epochs 5000 --weight_decay 5e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 7 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 32 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 5 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
    #--dataset texas --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 7 --split_graph False
#python train.py \
#    --lr 1e-3 --dropout 0.2 --cuda 7 --epochs 5000 --weight_decay 5e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 8 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
##    --use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 32 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 5 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset texas --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
#python train.py \
#    --lr 1e-3 --dropout 0.2 --cuda 7 --epochs 5000 --weight_decay 5e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 42 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 32 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 5 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset texas --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 42 --split_graph False
#wisconson
#python train.py \
#    --lr 1e-3 --dropout 0.2 --cuda 7 --epochs 5000 --weight_decay 5e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 7 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 32 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset wisconsin --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 7 --split_graph False
#python train.py \
#    --lr 1e-3 --dropout 0.2 --cuda 7 --epochs 5000 --weight_decay 5e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 10 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 32 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset wisconsin --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 10 --split_graph False
#python train.py \
#    --lr 1e-3 --dropout 0.2 --cuda 7 --epochs 5000 --weight_decay 5e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 25 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 32 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset wisconsin --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 25 --split_graph False
#chameleon
#python train.py \
    #--lr 1e-3 --dropout 0.02 --cuda 7 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 100 --seed 8 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 150 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 52 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 3 --KP_extent 0.66 --radius 1 --deformable False --linear_before 52 \
    #--dataset chameleon --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 8 --split_graph False
#python train.py \
    #--lr 1e-3 --dropout 0.02 --cuda 7 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 100 --seed 18 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 150 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 52 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 3 --KP_extent 0.66 --radius 1 --deformable False --linear_before 52 \
    #--dataset chameleon --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 18 --split_graph False
#python train.py \
    #--lr 1e-3 --dropout 0.02 --cuda 7 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 100 --seed 25 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 150 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 52 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 3 --KP_extent 0.66 --radius 1 --deformable False --linear_before 52 \
    #--dataset chameleon --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 25 --split_graph False
#squirral
python train.py \
    --lr 1e-3 --dropout 0.02 --cuda 7 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 100 --seed 10 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 150 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 36 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 36 \
    --dataset squirrel --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 10 --split_graph False
python train.py \
    --lr 1e-3 --dropout 0.02 --cuda 7 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 100 --seed 25 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 150 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 36 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 36 \
    --dataset squirrel --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 25 --split_graph False
python train.py \
    --lr 1e-3 --dropout 0.02 --cuda 7 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 100 --seed 42 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 150 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 36 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 36 \
    --dataset squirrel --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 42 --split_graph False
#film
python train.py \
    --lr 1e-3 --dropout 0.02 --cuda 7 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 100 --seed 8 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 150 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 3 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset film --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 8 --split_graph False
python train.py \
    --lr 1e-3 --dropout 0.02 --cuda 7 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 100 --seed 18 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 150 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 3 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset film --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 18 --split_graph False
python train.py \
    --lr 1e-3 --dropout 0.02 --cuda 7 --epochs 5000 --weight_decay 0 --optimizer radam --momentum 0.999 --patience 100 --seed 25 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 150 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt False --AggKlein True --corr 0 --nei_agg 2 --task nc --model BKNet --dim 64 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 3 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset film --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 25 --split_graph False