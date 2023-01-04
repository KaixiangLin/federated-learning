


python main.py --dataset=synthetic_logistic_regression_iclr --epsilon=0.9 --num_rounds=500000 --num_epochs=1 --number_user=10 --learning_rate=0.01 --is_decay=True --lrconst=1 --machine=gpu --batch_size=40 --num_iteration=50000 --model=logisticregression --optimizer=fedave --dimension=60 --adapt=0 --seed=0




python main.py --dataset=synthetic_logistic_regression_iclr300 --epsilon=0.9 --num_rounds=500000 --num_epochs=1 --number_user=10 --learning_rate=0.01 --is_decay=True --lrconst=1 --machine=gpu --batch_size=4 --num_iteration=50000 --model=logisticregression --optimizer=fedave --dimension=300 --adapt=0 --seed=0




## binarylogisticregression
d = 300, n = 49749

python main.py --dataset=w8a --epsilon=0.1 --num_rounds=500000 --num_epochs=1 --number_user=10 --learning_rate=1 --is_decay=True --lrconst=1554 --machine=gpu --batch_size=4 --num_iteration=50000 --model=binarylogisticregression --optimizer=fedave --dimension=300 --adapt=0 --seed=0 --regularization=1e-05


nohup python main.py --dataset=w8a --seed=0 --epsilon=0.13143317621654502 --num_rounds=500000 --num_epochs=1 --number_user=64 --learning_rate=15 --is_decay="" --lrconst=1 --machine=gpu --batch_size=4 --num_iteration=1000 --model=binarylogisticregression --optimizer=fedave --regularization=1e-05 --dimension=300 --adapt=0 > ~/tmp/nohup2.output 2>&1 &


python main.py --dataset=w8a --seed=0 --epsilon=0.13143317621654502 --num_rounds=500000 --num_epochs=1 --number_user=64 --learning_rate=1 --is_decay="" --lrconst=1 --machine=gpu --batch_size=4 --num_iteration=400 --model=binarylogisticregression --optimizer=fedave --regularization=1e-05 --dimension=300 --adapt=1


H = 4
python main.py --dataset=w8a --seed=0 --epsilon=0.13143317621654502 --num_rounds=500000 --num_epochs=1 --number_user=64 --learning_rate=1 --is_decay="" --lrconst=1 --machine=gpu --batch_size=4 --num_iteration=400 --model=binarylogisticregression --optimizer=fedave --regularization=1e-05 --dimension=300 --adapt=1


python main.py --dataset=w8a --seed=1 --epsilon=0.13143317621654502 --num_rounds=500000 --num_epochs=64 --number_user=256 --learning_rate=0.25 --is_decay=True --lrconst=0.5 --machine=gpu --batch_size=4 --num_iteration=80000 --model=binarylogisticregression --optimizer=fedave --regularization=1e-05 --dimension=300 --adapt=0

# regularization=0 convex smooth



# epsilon = 0
python main.py --dataset=w8a --seed=0 --epsilon=0 --num_rounds=500000 --num_epochs=1 --number_user=1024 --learning_rate=0.5 --is_decay=True --lrconst=64 --machine=gpu --batch_size=4 --num_iteration=100000 --model=binarylogisticregression --optimizer=fedave --regularization=1e-5 --dimension=300 --adapt=0



python main.py --dataset=linearregressionw8a --seed=0 --epsilon=0.02 --num_rounds=500000 --num_epochs=16 --number_user=32 --learning_rate=0.015625 --is_decay=True --lrconst=0.01 --machine=gpu --batch_size=4 --num_iteration=100000 --model=linearregression --optimizer=fedave --regularization=0 --dimension=300 --adapt=0

1/32 = 0.03125
1/64 = 0.015625
1/128 = 0.0078125
1/256 = 0.00390625
1/512 = 0.001953125
1/1024 = 0.0009765625


# nesterov sgd
python main.py --dataset=linearregressionw8a --seed=0 --epsilon=0.02 --num_rounds=500000 --num_epochs=4 --number_user=32 --learning_rate=0.015625 --is_decay=True --lrconst=0.1 --machine=gpu --batch_size=4 --num_iteration=100000 --model=linearregression --optimizer=fednesterov --regularization=0 --dimension=300 --adapt=0 --partial_participation=1


python main.py --dataset=w8a --seed=1 --epsilon=0.13143317621654502 --num_rounds=500000 --num_epochs=4 --number_user=32 --learning_rate=0.015625 --is_decay=True --lrconst=0.1 --machine=gpu --batch_size=4 --num_iteration=80000 --model=binarylogisticregression --optimizer=fednesterov --regularization=1e-05 --dimension=300 --adapt=0

python main.py --dataset=w8a --seed=1 --epsilon=0.13143317621654502 --num_rounds=500000 --num_epochs=4 --number_user=32 --learning_rate=0.125 --is_decay=True --lrconst=32 --machine=gpu --batch_size=4 --num_iteration=80000 --model=binarylogisticregression --optimizer=fednesterov --regularization=1e-05 --dimension=300 --adapt=0

# partial participation
python main.py --dataset=linearregressionw8a --seed=0 --epsilon=0.02 --num_rounds=500000 --num_epochs=1 --number_user=32 --learning_rate=0.015625 --is_decay=True --lrconst=0.1 --machine=gpu --batch_size=4 --num_iteration=100000 --model=linearregression --optimizer=fedave --regularization=0 --dimension=300 --adapt=0 --partial_participation=0.5








