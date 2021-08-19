export PYTHONPATH=$PYTHONPATH:'pwd'
export CUDA_VISIBLE_DEVICES=$1
python run/cifar/train.py --lr 1e-1 -a $2 -d cifar10 --conv-type fp