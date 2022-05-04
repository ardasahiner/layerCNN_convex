#!/bin/bash
module load python/3.9.0
module load py-numpy/1.20.3_py39
module load py-pytorch/1.8.1_py39
module load py-scipy/1.6.3_py39

nvidia-smi

if [ $6 -gt 0 ]
then
	if [ $7 -eq 0 ]
	then
		python3 cifar_convex.py --lr $1 $2 $3 $4 $5  --data_dir /home/users/sahiner/data --burer_monteiro --mse
	elif [ $7 -eq 1 ]
	then
		python3 cifar_convex.py --lr $1 $2 $3 $4 $5  --data_dir /home/users/sahiner/data --burer_monteiro --mse --hinge_loss --lambda_hinge_loss $8 $9 $10 $11 $12
	else
		python3 cifar_convex.py --lr $1 $2 $3 $4 $5  --data_dir /home/users/sahiner/data --burer_monteiro --mse --hinge_loss --lambda_hinge_loss $8 $9 $10 $11 $12 --squared_hinge
	fi
else
	if [ $7 -eq 0 ]
	then
		python3 cifar_convex.py --lr $1 $2 $3 $4 $5  --data_dir /home/users/sahiner/data --burer_monteiro
	elif [ $7 -eq 1 ]
	then
		python3 cifar_convex.py --lr $1 $2 $3 $4 $5  --data_dir /home/users/sahiner/data --burer_monteiro --hinge_loss --lambda_hinge_loss $8 $9 $10 $11 $12
	else
		python3 cifar_convex.py --lr $1 $2 $3 $4 $5  --data_dir /home/users/sahiner/data --burer_monteiro  --hinge_loss --lambda_hinge_loss $8 $9 $10 $11 $12 --squared_hinge
	fi
fi

