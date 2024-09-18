#cornell
#for kernel_size in {2..9}
#do
    #python train.py \
    #--lr 0.0005 --dropout 0 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 50 --seed 23 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 64 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 1.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    #--dataset cornell --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 23 --split_graph False
    #python train.py \
    #--lr 0.0005 --dropout 0 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 50 --seed 8 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 64 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 1.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    #--dataset cornell --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
    #python train.py \
    #--lr 0.0005 --dropout 0 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 50 --seed 10 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    #--use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 64 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 1.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    #--dataset cornell --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 10 --split_graph False
#done
#texas
#for kernel_size in {2..9}
#do
#    python train.py \
#    --lr 0.0005 --dropout 0.1 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 100 --seed 28 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset texas --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 28 --split_graph False
#    python train.py \
#    --lr 0.0005 --dropout 0.1 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 100 --seed 8 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset texas --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
#    python train.py \
#    --lr 0.0005 --dropout 0.1 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 100 --seed 7 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset texas --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 7 --split_graph False
#done
#wisconsin
#for kernel_size in {2..9}
#do
#    python train.py \
#    --lr 0.0005 --dropout 0.1 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 50 --seed 12 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset wisconsin --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 12 --split_graph False
#    python train.py \
#    --lr 0.0005 --dropout 0.1 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 50 --seed 11 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset wisconsin --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 11 --split_graph False
#    python train.py \
#    --lr 0.0005 --dropout 0.1 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 50 --seed 22 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset wisconsin --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 22 --split_graph False
#done
#chameleon
#for kernel_size in {2..9}
#do
#    python train.py \
#    --lr 0.005 --dropout 0.45 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 100 --seed 35 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 0.8 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
#    --dataset chameleon --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 35 --split_graph False
#    python train.py \
#    --lr 0.005 --dropout 0.45 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 100 --seed 30 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 0.8 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
#    --dataset chameleon --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 30 --split_graph False
#    python train.py \
#    --lr 0.005 --dropout 0.45 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 100 --seed 13 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 0.8 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
#    --dataset chameleon --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 13 --split_graph False
#done
#squirral
#for kernel_size in {2..9}
#do
#    python train.py \
#    --lr 0.005 --dropout 0.2 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 24 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
#    --dataset squirrel --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 24 --split_graph False
#    python train.py \
#    --lr 0.005 --dropout 0.2 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 42 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
#    --dataset squirrel --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 42 --split_graph False
#    python train.py \
#    --lr 0.005 --dropout 0.2 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 28 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
#    --dataset squirrel --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 28 --split_graph False
#done
#actor
#for kernel_size in {2..9}
#do
#    python train.py \
#    --lr 1e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-3 --optimizer radam --momentum 0.999 --patience 100 --seed 8 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 250 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset film --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
#    python train.py \
#    --lr 1e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-3 --optimizer radam --momentum 0.999 --patience 100 --seed 42 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 250 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset film --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 42 --split_graph False
#    python train.py \
#    --lr 1e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-3 --optimizer radam --momentum 0.999 --patience 100 --seed 25 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 250 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset film --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 25 --split_graph False
#done

################corr=0############
#cornell
python train.py \
    --lr 0.0005 --dropout 0 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 50 --seed 23 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 64 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 1.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 7 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset cornell --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 23 --split_graph False
    python train.py \
    --lr 0.0005 --dropout 0 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 50 --seed 8 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 64 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 1.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 7 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset cornell --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
    python train.py \
    --lr 0.0005 --dropout 0 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 50 --seed 10 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 64 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 1.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 7 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    --dataset cornell --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 10 --split_graph False
#texas
python train.py \
    --lr 0.0005 --dropout 0.1 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 100 --seed 28 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 5 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
    --dataset texas --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 28 --split_graph False
python train.py \
    --lr 0.0005 --dropout 0.1 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 100 --seed 8 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 5 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
    --dataset texas --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
python train.py \
    --lr 0.0005 --dropout 0.1 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 100 --seed 7 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 100 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 5 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
    --dataset texas --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 7 --split_graph False
#wisconson
#python train.py \
#    --lr 0.0005 --dropout 0.1 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 50 --seed 12 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset wisconsin --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 12 --split_graph False
#python train.py \
#    --lr 0.0005 --dropout 0.1 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 50 --seed 11 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset wisconsin --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 11 --split_graph False
#python train.py \
#    --lr 0.0005 --dropout 0.1 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 50 --seed 22 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 200 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
#    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
#    --dataset wisconsin --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 22 --split_graph False
#chameleon
python train.py \
    --lr 0.005 --dropout 0.45 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 100 --seed 35 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 0.8 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 8 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset chameleon --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 35 --split_graph False
python train.py \
    --lr 0.005 --dropout 0.45 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 100 --seed 30 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 0.8 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 8 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset chameleon --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 30 --split_graph False
python train.py \
    --lr 0.005 --dropout 0.45 --cuda 1 --epochs 5000 --weight_decay 0.0001 --optimizer radam --momentum 0.999 --patience 100 --seed 13 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 0.8 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 8 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset chameleon --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 13 --split_graph False
#squiral
python train.py \
    --lr 0.005 --dropout 0.2 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 24 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset squirrel --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 24 --split_graph False
python train.py \
    --lr 0.005 --dropout 0.2 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 42 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset squirrel --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 42 --split_graph False
python train.py \
    --lr 0.005 --dropout 0.2 --cuda 1 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 100 --seed 28 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 1000 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2 --pretrained_embeddings None --pos_weight 0 --num_layers 3 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset squirrel --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 28 --split_graph False
#actor
python train.py \
    --lr 1e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-3 --optimizer radam --momentum 0.999 --patience 100 --seed 8 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 250 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
    --dataset film --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 8 --split_graph False
python train.py \
    --lr 1e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-3 --optimizer radam --momentum 0.999 --patience 100 --seed 42 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 250 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
    --dataset film --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 42 --split_graph False
python train.py \
    --lr 1e-3 --dropout 0.05 --cuda 1 --epochs 5000 --weight_decay 1e-3 --optimizer radam --momentum 0.999 --patience 100 --seed 25 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 250 --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 0 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 4 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
    --dataset film --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 25 --split_graph False