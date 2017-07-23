# DeepEmbeddedClustering
chainer implementation of Deep Embedded Clustering([Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/abs/1511.06335))  
In this code, we use MNIST as training data.

## Requirement
- Chainer 2.0.0
- Cupy 1.0.0
	- if use GPU
- scikit-learn 0.18.1

## Running
### Pretraining
```shell
$ python pretraining.py --gpu=0 --seed=0 
```

`--gpu=0` turns on GPU. If you turn off GPU, use `--gpu=-1` or remove `--gpu` option. `--seed=0` means random seed.  

### Training model
```shell
$ python main.py --gpu=0 --seed=0 --model_seed=0 --cluster=10 
```
`--gpu` and `--seed` means same as before. `--model_seed` is seed number when pretraing.  
Every five iteration, save embedding result in directory like `modelseed0_seed0/`.  
I used t-SNE and compress embedding vector to 2-dim. And I saved embedding result of 500 data as scatter plot. 

## Reference
Junyuan Xie, Ross Girshick, Ali Farhadi, "Unsupervised Deep Embedding for Clustering Analysis" [https://arxiv.org/abs/1511.06335](https://arxiv.org/abs/1511.06335)
