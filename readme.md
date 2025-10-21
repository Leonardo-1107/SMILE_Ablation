# For SMILE ablation study


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