import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from monai.losses import DiceCELoss
seg_loss_fn = DiceCELoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True,
    lambda_dice=1.0,
    lambda_ce=1.0,
    reduction="sum"
)

from utils import center_crop





def get_segmentation(seg_model, estimated_image_for_nnunet, mini_batch=False):
    """
    Get the segmentation logits from reconstructed CT.
    The CT needs recover to HU.
    """
    
    # mini-batch to reduce memory used.
    if mini_batch:
        pred_logits = []
        for i in range(estimated_image_for_nnunet.size(0)):
            mini_slice = estimated_image_for_nnunet[i:i+1].float()
            pred = seg_model(mini_slice)
            pred_logits.append(pred)
        pred_logits = torch.cat(pred_logits, dim=0)
    else:
        pred_logits = seg_model(estimated_image_for_nnunet.float())
    
    return pred_logits
    

def get_classification(phase_classifier, estimated_image, repeat_channel=False):
    b, c, h, w = estimated_image.shape
    
    # NOTE: classification model, In-channel=3
    if repeat_channel:# timm requires processed
        classify_logits = phase_classifier(estimated_image.float().reshape(b*c, 1, h, w).repeat(1, 3, 1, 1)) 
    else:
        estimated_image_01 = (estimated_image + 1000.0) /2000.0
        classify_logits = phase_classifier(estimated_image_01.float())#MONAI, requires (0, 1)
    
    
    return classify_logits


def unchanged_region_loss(batch, estimated_image, HU_threshold=800):
    """
    Pixel-wise loss on the unchanged region (air and bone) between cond image and translated image.
    estimated_image shall be in HU domain.
    """
    unchanged_mask = batch["unchanged_mask"].to(estimated_image.device)  # 1 where unchanged, 0 where changed
    estimated_image_unchanged_mask = ((estimated_image < -HU_threshold) | (estimated_image > HU_threshold))
    estimated_image_01 = (estimated_image+1000.0) / 2000.0 
    cond_image_01 = batch["cond_pixel_values_original"].to(estimated_image.device)
    loss = F.l1_loss(cond_image_01 * unchanged_mask.float(), estimated_image_01 * estimated_image_unchanged_mask.float())
    return loss


def organ_mean_hu(x, mask, organ_id):
    """
    Organ-volume averaged HU value.
    x is preferablly in [0, 1] domain
    """

    organ_mask = (mask == organ_id)
    organ_area = organ_mask.sum().clamp(min=1.0)
    
    organ_mask = organ_mask.reshape(x.shape)
    return (x * organ_mask.float()).sum() / organ_area


def HU_avg_loss(vae, batch, estimated_image, pred_logits, crop_to_center=False):
    """
    Compute the organ-wise average HU value MSE loss for translated image
    and its ground truth image, with higher penalties on specific organs.
    
    Thanks to the loose constraints design, pixel-level registration is not required.

    penalty_weight_dict:
        the adjustable penalty weight dict, set the weight for certer organ MSE errors
        based on clinical knowledge
    """

    penalty_weight_dict = {
        1: 100,   # aorta
        10: 100,  # kidney left
        11: 100,  # kidney right
        12: 100,  # liver
        17: 10,   # stomach
        18: 10,   # spleen
    }

    # the gt pixel values, in [0, 1] range
    pixel_values = batch["pixel_values"].squeeze(1).to(vae.device)
    gt_mask = batch["mask_values"].squeeze(1).to(vae.device).long()
    estimated_image_01 = (estimated_image + 1000.0) / 2000.0
    translated_values = estimated_image_01.squeeze(1)
    pred_mask = (torch.argmax(pred_logits, dim=1).to(vae.device) != 0).squeeze(1).long()


    # crop to only focus on center
    if crop_to_center:
        pixel_values = center_crop(pixel_values)
        translated_values = center_crop(translated_values)
        gt_mask = center_crop(gt_mask)
        pred_mask = center_crop(pred_mask).reshape(gt_mask.shape)
    
    # exclude background, organ-wise
    gt_ids = torch.unique(gt_mask)
    gt_ids = gt_ids[gt_ids != 0]  
    pred_ids = torch.unique(pred_mask)

    if len(gt_ids) == 0:# no organ in gt mask
        return torch.tensor(0.0, device=vae.device)

    organ_losses = []
    for organ_id in gt_ids:
        organ_id = organ_id.item()
        gt_hu_val = organ_mean_hu(pixel_values, gt_mask, organ_id)

        if organ_id in pred_ids:
            pred_hu_val = organ_mean_hu(translated_values, pred_mask, organ_id)
        else:
            pred_hu_val = torch.tensor(0.0, device=vae.device)

        # apply penalty if organ in dict, else weight = 1
        organ_loss = (pred_hu_val - gt_hu_val) ** 2
        weight = penalty_weight_dict.get(organ_id, 1.0)
        organ_losses.append(weight * organ_loss)

    # aggregate losses
    HU_Loss = torch.mean(torch.stack(organ_losses))
    HU_Loss = torch.clamp(HU_Loss, max=1e4)

    return HU_Loss


def segmentation_loss(batch, pred_logits, estimated_image, use_gt_mask=False):  
    """
    Via the MONAI-DiceCELoss.

    estimated_image: (B, C, H, W), shall be in HU domain.
    """
    b, c, h, w = estimated_image.shape
    N = b * c

    if use_gt_mask:
        cond_mask = batch["cond_mask_values"].reshape(N, h, w).long()
    else:
        cond_mask = batch["cond_mask_pred"] # already in (N, 1, H, W) due to the form of 2D nnUNet segmentation
        if cond_mask.ndim == 4 and cond_mask.shape[1] == 1:
            cond_mask = cond_mask.squeeze(1)
        cond_mask = cond_mask.long()
        
    seg_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="mean") # still CE
    eff_seg_loss = seg_loss_fn(pred_logits, cond_mask) 
    return eff_seg_loss



def classification_loss(classify_logits, gt_phase, repeat_channel=False):
    """phase-classfication CE loss"""
    
    if repeat_channel:# for old timm model version
        gt_phase = gt_phase.repeat(3)
    cls_loss_per = F.cross_entropy(classify_logits, gt_phase)
    eff_cls_loss = (cls_loss_per).mean()
    
    return eff_cls_loss


def cycle_mse_loss(batch, estimated_image, cycle_estimated_image):
    """
        Cycle Image MSE loss, pixel-wise
        
        estimated_image shall be in HU domain.
    """

    cycle_estimated_image_01 = ((cycle_estimated_image + 1000.0)/2000.0).to(estimated_image.device)# to [0, 1] range
    cycle_origin_pixel_values_01 = batch["cond_pixel_values_original"].to(estimated_image.device)# translation source
    cycle_loss = F.mse_loss(cycle_estimated_image_01, cycle_origin_pixel_values_01)
    return cycle_loss



#========================== Uncertainty Weighted Loss Module ==========================# 
class UncertaintyWeightedLoss(torch.nn.Module):
    """
    Uncertainty Weighted Multi-Task Loss, Kendall et al., 2018.
    https://arxiv.org/abs/1705.07115
    """
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):  # list of scalar losses
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss





#========================== General Step Loss Functions ==========================#

def step3_loss(batch, estimated_image, cycle_estimated_image, classify_logits, diffusion_loss, uc_area_lambda=1, cls_lambda=1e-2, cycle_lambda=1):
    """
    Loss function for step3, where the phase loss is added.
    Cycle process is also deployed to ensure structure consistancy.
    """

    b, c, h, w = estimated_image.shape
    unchanged_loss = unchanged_region_loss(batch, estimated_image)
    cls_loss = classification_loss(
        classify_logits, # repeat channel
        batch["gt_phase_id"].to(classify_logits.device))
    cycle_loss = cycle_mse_loss(batch, estimated_image, cycle_estimated_image)
    # print(f"UC area loss: {unchanged_loss.item()}")
    return diffusion_loss + uc_area_lambda*unchanged_loss + cls_lambda*cls_loss + cycle_lambda*cycle_loss 


def step4_loss(vae, batch, estimated_image, classify_logits, pred_logits, diffusion_loss, uc_area_lambda=1, cls_lambda=1e-2, seg_dsc_lambda=1e-2, hu_mse_lambda=1):
    """
    Controling the segmentation structure correctness and translated HU correlation.
    Formula:
        Loss = diffusion_loss + segmentation_loss + classification_loss + hu_average_loss
    """
    b, c, h, w = estimated_image.shape
    unchanged_loss = unchanged_region_loss(batch, estimated_image)
    cls_loss = classification_loss(classify_logits, batch["gt_phase_id"].to(classify_logits.device))
    seg_loss = segmentation_loss(batch, pred_logits, estimated_image)
    hu_avg_loss = HU_avg_loss(vae, batch, estimated_image, pred_logits)
    # print(f"Step4 Losses: Diffusion {diffusion_loss.item():.4f}, Cls {cls_loss.item():.4f}, Seg {seg_loss.item():.4f}, UC area: {unchanged_loss.item():.4f}, HU {hu_avg_loss.item():.4f}")
    
    return diffusion_loss + uc_area_lambda*unchanged_loss + cls_lambda*cls_loss + seg_dsc_lambda*seg_loss + hu_mse_lambda*hu_avg_loss


def step5_loss(vae, batch, estimated_image, cycle_estimated_image, classify_logits, pred_logits, diffusion_loss, auto_adjust=False, uc_area_lambda=1, cls_lambda=1e-2, seg_dsc_lambda=1, hu_mse_lambda=1, cycle_lambda=100):
    """
    Add Cycle Diffusion MSE Loss
    """
    fore_loss = step4_loss(vae, batch, estimated_image, classify_logits, pred_logits, diffusion_loss, uc_area_lambda, cls_lambda, seg_dsc_lambda, hu_mse_lambda)
    cycle_loss = cycle_mse_loss(batch, estimated_image, cycle_estimated_image)
    return fore_loss + cycle_lambda * cycle_loss


def step6_loss(
    vae, 
    batch, 
    estimated_image, 
    cycle_estimated_image, 
    classify_logits, pred_logits, 
    diffusion_loss, 
    uncertainty_loss_module=None, 
    uc_area_lambda=1, cls_lambda=1e-2, seg_dsc_lambda=1, hu_mse_lambda=100, cycle_lambda=100):
    """
    Add learnable parameters, to punish more on more obvious loss.
    """
    b, c, h, w = estimated_image.shape
    unchanged_loss = unchanged_region_loss(batch, estimated_image)
    cls_loss = classification_loss(classify_logits, batch["gt_phase_id"].to(classify_logits.device))
    seg_loss = segmentation_loss(batch, pred_logits, estimated_image)
    hu_avg_loss = HU_avg_loss(vae, batch, estimated_image, pred_logits)
    cycle_loss = cycle_mse_loss(batch, estimated_image, cycle_estimated_image)
    
    if uncertainty_loss_module is not None:
        # auto-adjust the classification, segmentation and HU losses
        composite_loss = uncertainty_loss_module([cls_loss, seg_loss, hu_avg_loss])
        total_loss = diffusion_loss + uc_area_lambda*unchanged_loss + cls_lambda * composite_loss + cycle_lambda * cycle_loss

        # add safety-check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return (# fall back
                diffusion_loss +
                uc_area_lambda * unchanged_loss +
                cls_lambda * cls_loss +
                seg_dsc_lambda * seg_loss +
                hu_mse_lambda * hu_avg_loss +
                cycle_lambda * cycle_loss
            )
        return total_loss
    
    else:
        total_loss = (
            diffusion_loss +
            uc_area_lambda * unchanged_loss +
            cls_lambda * cls_loss +
            seg_dsc_lambda * seg_loss +
            hu_mse_lambda * hu_avg_loss +
            cycle_lambda * cycle_loss
        )

    return total_loss





