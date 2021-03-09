## **Examples for Reproducing the Results**

CP:
```
python run_distshift_randomized_mp.py --method DQN_CP --num_explorers 8 --ignore_model 0 --layers_model 1 --signal_predict_action 1 --disable_bottleneck 0 --size_bottleneck 8
```

UP:
```
python run_distshift_randomized_mp.py --method DQN_CP --num_explorers 8 --ignore_model 0 --layers_model 1 --signal_predict_action 1 --disable_bottleneck 1
```

WM:

```
python run_distshift_randomized_mp.py --method DQN_WM --num_explorers 8 --ignore_model 0 --layers_model 1 --signal_predict_action 1 --disable_bottleneck 0 --size_bottleneck 8 --period_warmup 1000000
```

Dyna:
```
python run_distshift_randomized_mp.py --method DQN_WM --num_explorers 8 --ignore_model 0 --layers_model 1 --disable_bottleneck 0 --size_bottleneck 8 --learn_dyna_model 1
```

Dyna*:
```
python run_distshift_randomized_mp.py --method DQN_WM --num_explorers 8 --ignore_model 0 --layers_model 1 --disable_bottleneck 0 --size_bottleneck 8 --learn_dyna_model 0
```

NOSET:
```
python run_distshift_randomized_mp.py --method DQN_WM --num_explorers 8 --ignore_model 0 --layers_model 2 --len_hidden 256
```