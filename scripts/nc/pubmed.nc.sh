#python train.py --task nc --dataset pubmed --model HKPNet --dim 16 --lr 0.0005 --num-layers 2 --act relu --bias 1 --dropout 0.9 --weight-decay 1e-4 --manifold Lorentz --log-freq 5 --patience 1500 --linear-before 16 --cuda 0 --grad-clip 0.5 --kernel-size 2

#HKN_neoe Fine setting
#python train.py \
    #--lr 1e-3 --dropout 0.6 --cuda  --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 250 --seed 25 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    #--use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    #--dataset pubmed --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 25 --split_graph False

for kernel_size in {3..9}
do
    python train.py \
    --lr 1e-3 --dropout 0.6 --cuda 7 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 250 --seed 9 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset pubmed --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 9 --split_graph False
    python train.py \
    --lr 1e-3 --dropout 0.6 --cuda 7 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 250 --seed 42 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset pubmed --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 42 --split_graph False
    python train.py \
    --lr 1e-3 --dropout 0.6 --cuda 7 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 250 --seed 20 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    --use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 16 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size $kernel_size --KP_extent 0.66 --radius 1 --deformable False --linear_before 16 \
    --dataset pubmed --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 20 --split_graph False
done


#HKN_neoe try more seems to be better
#python train.py \
    #--lr 1e-3 --dropout 0.65 --cuda 7 --epochs 5000 --weight_decay 1e-4 --optimizer radam --momentum 0.999 --patience 250 --seed 42 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 500 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    #--use_geoopt True --AggKlein False --corr 1 --nei_agg 0 --task nc --model HKPNet --dim 32 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 32 \
    #--dataset pubmed --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 42 --split_graph False


#BKNet
#python train.py \
    #--lr 1e-3 --dropout 0.2 --cuda 7 --epochs 5000 --weight_decay 1e-5 --optimizer radam --momentum 0.999 --patience 50 --seed 7 --log_freq 10 --eval_freq 10 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq 150 --gamma 0.5 --print_epoch True --grad_clip 0.5 --min_epochs 100 \
    #--use_geoopt False --AggKlein True --corr 1 --nei_agg 0 --task nc --model BKNet --dim 12 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 3 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 2 --KP_extent 0.66 --radius 1 --deformable False --linear_before 12 \
    #--dataset pubmed --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 5 --split_seed 7 --split_graph False
