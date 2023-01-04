

### A Unified Linear Speedup Analysis of Stochastic FedAvg and Nesterov Accelerated FedAvg


This repository host the code base for the following paper:

> A Unified Linear Speedup Analysis of Stochastic FedAvg and Nesterov Accelerated FedAvg </br>
Zhaonan Qu, Kaixiang Lin, Zhaojian Li, Jiayu Zhou, Zhengyuan Zhou </br>
[Arxiv](https://arxiv.org/abs/2007.05690)

### Run 

Example command:
```
python main.py --dataset=w8a --epsilon=0.1 --num_rounds=500000 --num_epochs=1 --number_user=10 --learning_rate=1 --is_decay=True --lrconst=1554 --machine=gpu --batch_size=4 --num_iteration=50000 --model=binarylogisticregression --optimizer=fedave --dimension=300 --adapt=0 --seed=0 --regularization=1e-05
```



## Reference
If you find this work helpful in your research, please consider citing the following paper. The bibtex are listed below:
```
Qu, Z., Lin, K., Li, Z., Zhou, J. and Zhou, Z., 2020. A Unified Linear Speedup Analysis of Stochastic FedAvg and Nesterov Accelerated FedAvg. arXiv preprint arXiv:2007.05690.
```