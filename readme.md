# For SMILE ablation study

This repository contains the ablation study for the SMILE system, a CT enhancement framework that enriches single-phase CT scans with multi-phase contrast information.

Here, we systematically remove specific components of the loss function to evaluate their individual contributions to overall translation quality and diagnostic performance.

contact:
```
Kuma:

    ☎️：+1 4103250656
    📧：jliu452@jh.edu
```
## Hardware requirements
* local storage > 100 GB
* GPU VARM > 50 GB


# Quick start
```
conda activate smile
bash train_x.sh

x for:
    train_1.sh  -> without the phase classification loss
    train_2.sh  -> without the organ-wise HU loss
    train_3.sh  -> without the segmentation mask loss
    train_4.sh  -> without the cycle consistency loss
    train_5.sh  -> without the boundary loss
```

## Create environment
```
conda create -n smile python=3.11
conda activta smile
pip install -r requirements.txt
```


## Download the autoencoder checkpoints
```
huggingface-cli download MitakaKuma/SMILE --include="vae/*" --local-dir ./autoencoder --local-dir-use-symlinks False
```

## Download the diffusion model checkpoints
```
huggingface-cli download MitakaKuma/SMILE --include="diffusion/*" --local-dir="./"
```

## Download the segmentation nnunet
```
huggingface-cli download MitakaKuma/SMILE --include="segmenter/*" --local-dir="./"
```

## Download the monai classifier
```
huggingface-cli download MitakaKuma/SMILE --include="classifier/*" --local-dir="./"
```


## Download dataset (Dataset905) *(~ 38GB)*
```
huggingface-cli download MitakaKuma/Dataset905 --repo-type dataset --include "Dataset905.zip" --local-dir "./"
```

where the data are all already in `.h5` form.