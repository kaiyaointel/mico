### download CIFAR-10 kit and dataset
```
python download_cifar.py
```
This will download kit as ```cifar10.zip``` and download dataset into ```./cifar10_data```.

### unzip kit
```
unzip cifar10.zip
```
This will unzip ```cifar10.zip``` into ```./cifar10```

### save attack dataset
```
python save_attack_dataset.py
```
This will save a file "features" and a file "memberships" as attack dataset for attack model training.

### CPU set-ups
```
conda install -y mkl mkl-include jemalloc
```
```
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
```

### train attack model
```
numactl --cpunodebind=0 --membind=0 python train_attack_model.py
```

### run attack
```
python run_attack.py
```
