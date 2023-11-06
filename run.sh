python3 image_source.py  --trte val  --net resnet50 --lr 1e-2 --dset office-home --max_epoch 50 --s 0 --seed 3 

python3 home_rce.py  --s 0  --max_epoch 20 --interval 20 --lr_gamma 0.0 --seed 3 

python3 divideHome.py   --s 0 --max_epoch 10  --lr 1e-3  --lr_gamma 10.0  --seed 3 

