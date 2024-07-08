#Run via HKNet
#python train.py --task gc --dataset PTC --model HKPNet  --normalize-feats 1 --log-freq 5   --epochs 5000  --patience 1000  --lr 0.005  --dim 16 --num-layers 6 --batch-size 32 --val-prop 0.1 --test-prop 0 --cuda 0 --kernel-size 4
#python train.py \
    #--lr 1e-6 --dropout 0.2 --cuda 4 --epochs 1000 --weight_decay 1e-3 --optimizer radam --momentum 0.999 --patience 13 --seed 10 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq None --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 300 \
    #--use_geoopt True --AggKlein False --corr 0 --task gc --model HKPNet --dim 64 --manifold Lorentz --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 4 --KP_extent 0.66 --radius 1 --deformable False --linear_before 64 \
    #--dataset PTC --batch_size 64 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 10 --split_graph False

#Run via BKN
python train.py \
    --lr 1e-3 --dropout 0.3 --cuda 4 --epochs 1000 --weight_decay 1e-3 --optimizer radam --momentum 0.999 --patience 13 --seed 13 --log_freq 2 --eval_freq 2 --save 1 --save_dir None --sweep_c 0 --lr_reduce_freq None --gamma 0.5 --print_epoch True --grad_clip None --min_epochs 300 \
    --use_geoopt False --AggKlein True --corr 1 --task gc --model BKNet --dim 32 --manifold PoincareBall --c 1.0 --r 2.0 --t 1.0 --margin 2.0 --pretrained_embeddings None --pos_weight 0 --num_layers 2 --bias 1 --act relu --n_heads 4 --alpha 0.2 --double_precision 1 --use_att 0 --local_agg 0 --kernel_size 6 --KP_extent 0.7 --radius 1 --deformable False --linear_before 32 \
    --dataset PTC --batch_size 128 --val_prop 0.05 --test_prop 0.1 --use_feats 1 --normalize_feats 1 --normalize_adj 1 --split_seed 13 --split_graph False

