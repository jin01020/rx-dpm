# RX-Euler
This is the codebase for RX-Euler.
This repository is based on [NVlabs/edm](https://github.com/NVlabs/edm).

## Dependencies
We share the environment of the code by docker.
```
docker pull snucvlab/ogdm:edm
```
You can also use Dockerfile provided by [NVlabs/edm](https://github.com/NVlabs/edm).

## Pre-trained models

We use the pre-trained models provided by [NVlabs/edm](https://github.com/NVlabs/edm).

## Sampling 
```
torchrun --standalone --nproc_per_node=[N] generate.py --seeds=[0-n] --batch=[B] --out=[log_dir_root] --network=[model_to_sample] --solver=[euler/heun] --steps=[steps] --frequency=[k] --save_image=[True/False]
```
### Heun's method (NFE = 2*steps-1)
```
torchrun --standalone --nproc_per_node=4 generate.py --seeds=0-49999 --batch=512 --network=PATH_TO_NETWORK --solver=euler --steps=8 
```
### Euler method (NFE = steps)
```
torchrun --standalone --nproc_per_node=4 generate.py --seeds=0-49999 --batch=512 --network=PATH_TO_NETWORK --steps=15 
```
### RX-Euler (NFE = steps) extrapolation every k=2 steps
```
torchrun --standalone --nproc_per_node=4 generate.py --seeds=0-49999 --batch=512 --network=PATH_TO_NETWORK --steps=15 --frequency=2
```
### Save format
Generated images are saved as *.npz by default. To save individual images as *.png, 

```
torchrun --standalone --nproc_per_node=4 generate.py --seeds=0-49999 --batch=512 --network=PATH_TO_NETWORK --steps=15 --frequency=2 --save_image True
```

## Calculating FID

To compute Fr&eacute;chet inception distance (FID) for a given model and sampler, first generate 50,000 random images and then compare them against the dataset reference statistics using `fid.py`.
The reference statistics are provided by [NVlabs/edm](https://github.com/NVlabs/edm).

```
torchrun --standalone --nproc_per_node=1 fid.py calc --images=euler15_2.npz \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
```