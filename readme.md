# For SMILE ablation study

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


## Download dataset (Dataset905)
```
huggingface-cli download MitakaKuma/Dataset905 --repo-type dataset --include "Dataset905.zip" --local-dir "./"
```