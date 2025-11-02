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
* local storage > 200 GB
* GPU VARM > 50 GB, A100 / H100 preferred


# Quick start


## ⚠️special parameter in `train.sh` example:

```bash
# Output location
export OUTPUT_DIR="./logs/Exp1_1"   # Each experiment should have a unique folder name

# Resume from a previous checkpoint
--resume_from_checkpoint=$CHECKPOINT_DIR \

# (Optional) Start from a specific step
--init_global_step=$INIT_STEP \
```
<mark>Important:</mark> If `--init_global_step` is not set, the script will automatically continue from the step recorded in the checkpoint.
For example, if you resume from `--resume_from_checkpoint="./logs/Exp1_1/checkpoint-40000"`, then training will automatically restart from step 40,000.

<mark>Important:</mark> The checkpoints are expected to be saved every `checkpointing_steps`, which is set to `10,000` by default.


**Experiment 1**
To delete the specially designed loss, respectively. 5 experiments in total.
```
conda activate smile
bash train_x.sh

x for:
    train_1.sh  -> without the phase classification loss (clinical)
    train_2.sh  -> without the organ-wise HU loss (clinial)
    train_3.sh  -> without the segmentation mask loss (structure)
    train_4.sh  -> without the cycle consistency loss (intensity)
    train_5.sh  -> without the boundary loss (clinical)
```

**Experiment 2**
Delete the grouped 2 losses at one time, 10 experiment in total.

```
conda activate smile
bash train_xx.sh

xx for:
    train_12.sh  -> without the phase classification loss + organ-wise HU loss 
    train_13.sh  -> without the phase classification loss + segmentation mask loss
    train_14.sh  -> without the phase classification loss + cycle consistency loss
    train_15.sh  -> without the phase classification loss + boundary loss

    train_23.sh  -> without the organ-wise HU loss + segmentation mask loss
    train_24.sh  -> without the organ-wise HU loss + cycle
    train_25.sh  -> without the organ-wise HU loss + boundary loss

    train_34.sh  -> without the segmentation mask loss + cycle consistency loss 
    train_35.sh  -> without the boundary loss + segmentation loss

    train_45.sh  -> without the boundary + cycle

```
~~train_125.sh -> without the phase classification loss + organ-wise HU loss + boundary loss~~

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