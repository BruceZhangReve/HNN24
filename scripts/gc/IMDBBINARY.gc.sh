python train.py --task gc --dataset IMDBBINARY --model HKPNet  --normalize-feats 1 --log-freq 5   --epochs 5000  --patience 1000  --lr 0.005  --dim 16 --num-layers 6 --batch-size 32 --val-prop 0.1 --test-prop 0 --cuda 0 --kernel-size 4