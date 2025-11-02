# For ablation study
# delete the classifier loss

export nnUNet_raw=/projects/bodymaps/jliu452/nnuet/nnUNet_raw
export nnUNet_preprocessed=/projects/bodymaps/jliu452/nnuet/nnUNet_preprocessed
export nnUNet_results=/projects/bodymaps/jliu452/nnuet/nnUNet_results

export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="./autoencoder/vae"
export TRAINED_UNET_NAME="./diffusion"
export SEG_MODEL_NAME="./segmenter/nnUNetTrainer__nnUNetResEncUNetLPlans__2d"
export CLS_MODEL_NAME="./classifier/monai_cls.pth" 

# Temporary path with soft link
export TRAIN_DATA_DIR="./Dataset905" 
export TRAIN_DATA_SHAPE_EXCEL_DIR="./dataset_xinze2/step3-shapes-filtered.csv"

# output location
export OUTPUT_DIR="./logs/Exp2_15"

# without classifier loss
accelerate launch --mixed_precision="no" --num_processes=1 train_text_to_image.py \
  --sd_model_name_or_path=$SD_MODEL_NAME \
  --finetuned_vae_name_or_path=$FT_VAE_NAME \
  --pretrained_unet_name_or_path=$TRAINED_UNET_NAME \
  --seg_model_path=$SEG_MODEL_NAME \
  --cls_model_path=$CLS_MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --dataset_shape_file=$TRAIN_DATA_SHAPE_EXCEL_DIR \
  --output_dir=$OUTPUT_DIR \
  --vae_loss="l1" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=2 \
  --max_train_steps=100_000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --validation_steps=1000 \
  --checkpointing_steps=10000 \
  --checkpoints_total_limit=3 \
  --warmup_end_df_only=2000 \
  --warmup_end_add_cls=10000 \
  --warmup_end_add_cls_seg_hu=15000 \
  --warmup_end_add_cycle=96000 \
  --uc_area_loss_weight=0 \
  --cls_loss_weight=0 \
  --seg_loss_weight=1e-3 \
  --hu_loss_weight=1e-2 \
  --cycle_loss_weight=10.0 \
  --skip_validation \

 