import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -------- helpers --------

import torch

def iou_3d_rotated_metric(box_pred, box_true):
    """
    3D IoU for rotated 3D boxes in METRIC camera frame.

    Args:
        box_pred: (N, 7) tensor [x, y, z, w, l, h, yaw]
        box_true: (N, 7) tensor [x, y, z, w, l, h, yaw]
            - x: lateral (right +), in meters
            - y: vertical (down +), in meters
            - z: depth (forward +), in meters
            - w: width along x, meters
            - l: length along z, meters
            - h: height along y, meters
            - yaw: rotation around camera y-axis, radians
    Returns:
        iou3d: (N,) tensor, IoU in [0,1]
    """

    # unpack
    x1, y1, z1 = box_pred[:, 0], box_pred[:, 1], box_pred[:, 2]
    w1 = torch.clamp(box_pred[:, 3], min=0.1)
    l1 = torch.clamp(box_pred[:, 4], min=0.1)
    h1 = torch.clamp(box_pred[:, 5], min=0.1)
    yaw1 = box_pred[:, 6]

    x2, y2, z2 = box_true[:, 0], box_true[:, 1], box_true[:, 2]
    w2 = torch.clamp(box_true[:, 3], min=0.1)
    l2 = torch.clamp(box_true[:, 4], min=0.1)
    h2 = torch.clamp(box_true[:, 5], min=0.1)
    yaw2 = box_true[:, 6]

    # ----- BEV overlap (x–z) using oriented-rectangle approximation -----
    # center offsets
    dx = x2 - x1
    dz = z2 - z1

    # rotate offset into box1's local frame (around y)
    cos1 = torch.cos(-yaw1)
    sin1 = torch.sin(-yaw1)
    dx_local = cos1 * dx - sin1 * dz
    dz_local = sin1 * dx + cos1 * dz

    # half-sizes
    hw1, hl1 = w1 / 2.0, l1 / 2.0
    hw2, hl2 = w2 / 2.0, l2 / 2.0

    # relative yaw between boxes
    d_yaw = yaw2 - yaw1
    cos_d = torch.cos(d_yaw).abs()
    sin_d = torch.sin(d_yaw).abs()

    # effective extents of box2 in box1's frame
    # (standard rectangle-rotation envelope)
    eff_hw2 = hw2 * cos_d + hl2 * sin_d
    eff_hl2 = hw2 * sin_d + hl2 * cos_d

    # BEV intersection in x–z plane
    inter_w = (hw1 + eff_hw2 - dx_local.abs()).clamp(min=0.0)
    inter_l = (hl1 + eff_hl2 - dz_local.abs()).clamp(min=0.0)
    inter_area = inter_w * inter_l

    area1 = w1 * l1
    area2 = w2 * l2
    union_area = area1 + area2 - inter_area + 1e-6
    bev_iou = (inter_area / union_area).clamp(min=0.0, max=1.0)

    # ----- height overlap (y) -----
    y1_min, y1_max = y1 - h1 / 2.0, y1 + h1 / 2.0
    y2_min, y2_max = y2 - h2 / 2.0, y2 + h2 / 2.0

    inter_min_y = torch.max(y1_min, y2_min)
    inter_max_y = torch.min(y1_max, y2_max)
    inter_h = (inter_max_y - inter_min_y).clamp(min=0.0)

    # ----- 3D IoU -----
    inter_vol = inter_area * inter_h
    vol1 = area1 * h1
    vol2 = area2 * h2
    union_vol = vol1 + vol2 - inter_vol + 1e-6

    iou3d = (inter_vol / union_vol).clamp(min=0.0, max=1.0)
    return iou3d

def iou_bev_rotated_metric(box_pred, box_true):
    """
    3D IoU for rotated 3D boxes in METRIC camera frame.

    Args:
        box_pred: (N, 7) tensor [x, y, z, w, l, h, yaw]
        box_true: (N, 7) tensor [x, y, z, w, l, h, yaw]
            - x: lateral (right +), in meters
            - y: vertical (down +), in meters
            - z: depth (forward +), in meters
            - w: width along x, meters
            - l: length along z, meters
            - h: height along y, meters
            - yaw: rotation around camera y-axis, radians
    Returns:
        iou3d: (N,) tensor, IoU in [0,1]
    """

    # unpack
    x1, y1, z1 = box_pred[:, 0], box_pred[:, 1], box_pred[:, 2]
    w1 = torch.clamp(box_pred[:, 3], min=0.1)
    l1 = torch.clamp(box_pred[:, 4], min=0.1)
    h1 = torch.clamp(box_pred[:, 5], min=0.1)
    yaw1 = box_pred[:, 6]

    x2, y2, z2 = box_true[:, 0], box_true[:, 1], box_true[:, 2]
    w2 = torch.clamp(box_true[:, 3], min=0.1)
    l2 = torch.clamp(box_true[:, 4], min=0.1)
    h2 = torch.clamp(box_true[:, 5], min=0.1)
    yaw2 = box_true[:, 6]

    # ----- BEV overlap (x–z) using oriented-rectangle approximation -----
    # center offsets
    dx = x2 - x1
    dz = z2 - z1

    # rotate offset into box1's local frame (around y)
    cos1 = torch.cos(-yaw1)
    sin1 = torch.sin(-yaw1)
    dx_local = cos1 * dx - sin1 * dz
    dz_local = sin1 * dx + cos1 * dz

    # half-sizes
    hw1, hl1 = w1 / 2.0, l1 / 2.0
    hw2, hl2 = w2 / 2.0, l2 / 2.0

    # relative yaw between boxes
    d_yaw = yaw2 - yaw1
    cos_d = torch.cos(d_yaw).abs()
    sin_d = torch.sin(d_yaw).abs()

    # effective extents of box2 in box1's frame
    # (standard rectangle-rotation envelope)
    eff_hw2 = hw2 * cos_d + hl2 * sin_d
    eff_hl2 = hw2 * sin_d + hl2 * cos_d

    # BEV intersection in x–z plane
    inter_w = (hw1 + eff_hw2 - dx_local.abs()).clamp(min=0.0)
    inter_l = (hl1 + eff_hl2 - dz_local.abs()).clamp(min=0.0)
    inter_area = inter_w * inter_l

    area1 = w1 * l1
    area2 = w2 * l2
    union_area = area1 + area2 - inter_area + 1e-6
    bev_iou = (inter_area / union_area).clamp(min=0.0, max=1.0)

    return bev_iou

def mean_abs_relative_error(pred, target, eps=1e-6, reduction="mean"):
    # avoid division by zero
    denom = target.abs().clamp_min(eps)
    rel_err = (pred - target).abs() / denom

    if reduction == "mean":
        return rel_err.mean()
    elif reduction == "sum":
        return rel_err.sum()
    else:
        return rel_err  # no reduction


def rotation_loss_depr(pred_yaw, true_yaw):
    """
    pred_yaw_norm, true_yaw_norm in [-1,1], representing yaw / π.
    """
    pred_yaw = pred_yaw 
    true_yaw = true_yaw 
    diff = pred_yaw - true_yaw
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    loss = 0.5*(1.0 - torch.cos(diff))
    return loss.mean()

def rotation_loss_current(pred_yaw, true_yaw):
    sin_pred = torch.sin(pred_yaw)
    cos_pred = torch.cos(pred_yaw)

    sin_true = torch.sin(true_yaw)
    cos_true = torch.cos(true_yaw)

    # cos(a-b)=cosa*cosb+sina*sinb
    cos_diff = cos_pred*cos_true + sin_pred*sin_true    # [-1,1]
    
    loss = 0.5*(1-cos_diff)
    return loss.mean()

def rotation_loss_centerpoint(pred_yaw, true_yaw):
    # Encode both as (sin, cos) pairs
    sin_pred = torch.sin(pred_yaw)
    cos_pred = torch.cos(pred_yaw)

    sin_true = torch.sin(true_yaw)
    cos_true = torch.cos(true_yaw)

    # L1 on each component separately, then sum — exactly as CenterPoint does
    loss = F.l1_loss(sin_pred, sin_true) + F.l1_loss(cos_pred, cos_true)
    return loss

def rotation_loss(pred_yaw, true_yaw):
    diff_2x = 2* (pred_yaw - true_yaw)
    
    loss = 0.5*(1-torch.cos(diff_2x))
    
    return loss.mean()



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, scale_factor=1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.scale_factor = scale_factor

    def forward(self, inputs, targets):
        # inputs: logits; targets: {0,1}
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha*targets + (1-self.alpha)*(1-targets)
        loss = self.scale_factor * alpha_t * (1-pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class SoftFocalLoss(nn.Module):
    """Focal Loss for soft targets as per CenterPoint paper"""
    def __init__(self, beta=4.0, gamma=2.0, alpha=1.0, scale_factor=1, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.scale_factor = scale_factor

    def forward(self, inputs, targets):
        # inputs: logits; targets: [0,1]
        y_hat = torch.sigmoid(inputs)

        eps = 1e-8

        center_mask = targets.eq(1.0).float()
        edge_mask = targets.lt(1.0).float()
        

        center_weight = targets
        edge_weight = (1-targets)**self.beta

        center_term = -center_weight * (1-y_hat)**self.gamma*torch.log(y_hat+eps)*center_mask
        edge_term = -edge_weight*y_hat**self.gamma*torch.log(1-y_hat+eps)*edge_mask
        
        loss = center_term + edge_term

        loss = self.scale_factor*loss

        if self.reduction == 'mean':
            return loss.mean() 
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        


def iou_3d_axis_aligned_metric(box_pred, box_true):
    """
    3D IoU for axis-aligned boxes in METRIC camera frame.
    box_*: (N,7) [x,y,z,w,l,h,yaw] with x,y,z,w,l,h in meters, yaw in radians (ignored here).
    """
    x1, y1, z1 = box_pred[:, 0], box_pred[:, 1], box_pred[:, 2]
    w1 = torch.clamp(box_pred[:, 3], min=0.1)
    l1 = torch.clamp(box_pred[:, 4], min=0.1)
    h1 = torch.clamp(box_pred[:, 5], min=0.1)

    x2, y2, z2 = box_true[:, 0], box_true[:, 1], box_true[:, 2]
    w2 = torch.clamp(box_true[:, 3], min=0.1)
    l2 = torch.clamp(box_true[:, 4], min=0.1)
    h2 = torch.clamp(box_true[:, 5], min=0.1)

    # half extents (assume l along x, w along z)
    hl1, hw1, hh1 = l1 / 2.0, w1 / 2.0, h1 / 2.0
    hl2, hw2, hh2 = l2 / 2.0, w2 / 2.0, h2 / 2.0

    # x overlap (length)
    x1_min, x1_max = x1 - hl1, x1 + hl1
    x2_min, x2_max = x2 - hl2, x2 + hl2
    inter_x = (torch.min(x1_max, x2_max) - torch.max(x1_min, x2_min)).clamp(min=0.0)

    # y overlap (height)
    y1_min, y1_max = y1 - hh1, y1 + hh1
    y2_min, y2_max = y2 - hh2, y2 + hh2
    inter_y = (torch.min(y1_max, y2_max) - torch.max(y1_min, y2_min)).clamp(min=0.0)

    # z overlap (width/depth)
    z1_min, z1_max = z1 - hw1, z1 + hw1
    z2_min, z2_max = z2 - hw2, z2 + hw2
    inter_z = (torch.min(z1_max, z2_max) - torch.max(z1_min, z2_min)).clamp(min=0.0)

    inter_vol = inter_x * inter_y * inter_z
    vol1 = w1 * l1 * h1
    vol2 = w2 * l2 * h2
    union_vol = vol1 + vol2 - inter_vol + 1e-6

    iou3d = (inter_vol / union_vol).clamp(min=0.0, max=1.0)
    return iou3d


class YOLO3DFusionLoss(nn.Module):
    def __init__(
        self,
        num_classes=12,
        iou_loss_weight=0.0,
        center_loss_weight=2.0,
        dims_loss_weight=2.0,
        cls_loss_weight=1.0,
        obj_loss_weight=0.2,
        rot_loss_weight=0.5,
        iou_pred_weight=2.0,
        scale_weights=[1.0, 1.0, 1.0],  # NEW: weights for P3, P4, P5
        class_freq_weights = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.iou_loss_weight = iou_loss_weight
        self.center_loss_weight = center_loss_weight
        self.dims_loss_weight = dims_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.obj_loss_weight = obj_loss_weight
        self.rot_loss_weight = rot_loss_weight
        self.iou_pred_weight = iou_pred_weight
        self.scale_weights = scale_weights  # NEW

        self.focal_loss = SoftFocalLoss(beta=4.0, gamma=2.0, alpha=1.0, scale_factor=1.0, reduction='sum')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l2 = nn.MSELoss(reduction='mean')
        self.last_losses = {}
        
        self.epoch_scale_stats = []
        self.scale_average = None

        # NEW: Class balancing weights (inverse frequency)
        if class_freq_weights is None:
            # Only foreground classes matter — background classes are never
            # used in classification loss (it only fires on positive cells)
            raw_freqs = [0, 0, 280380, 0, 0, 0, 222951, 493322, 128050, 0, 12617, 11859]
            
            # Total only over foreground classes (non-zero)
            foreground_total = sum(f for f in raw_freqs if f > 0)  # 1,148,779
            num_foreground = sum(1 for f in raw_freqs if f > 0)    # 6
            
            weights = []
            for f in raw_freqs:
                if f > 0:
                    # Inverse frequency, normalized so mean foreground weight = 1.0
                    weights.append(foreground_total / (num_foreground * f))
                else:
                    # Background classes: weight doesn't matter, set to 1.0
                    weights.append(1.0)
            
            self.class_weights = torch.tensor(weights).float()
            
            # Cap at 3.0 — motorcycle and bicycle have ~40× fewer samples than car,
            # but amplifying their gradient by 40× is noise not signal.
            # 3.0 gives them a meaningful boost without dominating.
            self.class_weights = self.class_weights.clamp(max=2.0)
            
            # Re-normalize so mean foreground weight = 1.0 after clamping
            fg_mask = torch.tensor([1 if f > 0 else 0 for f in raw_freqs]).bool()
            fg_mean = self.class_weights[fg_mask].mean()
            self.class_weights[fg_mask] = self.class_weights[fg_mask] / fg_mean
        else:
            self.class_weights = torch.tensor(class_freq_weights).float()
            
        print(f"Class weights: {self.class_weights.tolist()}")
        self.last_losses = {}

    def _apply_class_weight(self, cls_loss, cls_true_pos):
        """Apply per-class weighting to cross-entropy loss"""
        if len(cls_true_pos) == 0:
            return cls_loss
            
        # Get class indices as long tensor
        class_ids = cls_true_pos.long()
        
        # Weight each sample by its class weight
        weights = self.class_weights.to(class_ids.device)[class_ids]
        
        # Apply weights to loss (broadcast over batch)
        weighted_loss = cls_loss * weights
        
        return weighted_loss.mean()  # already normalized by num_pos in caller

    def forward(self, preds, targets):
        """
        NEW: Process all three scales
        
        preds: List of [pred_p3, pred_p4, pred_p5]
            pred_p3: (B, 1+num_classes+7, 128, 128)
            pred_p4: (B, 1+num_classes+7, 64, 64)
            pred_p5: (B, 1+num_classes+7, 32, 32)
            
        targets: List of [tgt_p3, tgt_p4, tgt_p5]
            Each with 'objectness', 'boxes', 'classes'
        """
        # NEW: Initialize combined losses
        total_loss = 0.0
        combined_losses = {
            'total': 0.0,
            'obj': 0.0,
            'cls': 0.0,
            'iou': 0.0,
            'center': 0.0,
            'dims': 0.0,
            'rot': 0.0,
            'iou_pred': 0.0
        }
        
        # NEW: Track per-scale statistics
        scale_stats = []
        scale_loss_dicts = []
        # NEW: Process each scale
        for scale_idx, (pred, tgt, scale_weight) in enumerate(zip(preds, targets, self.scale_weights)):
            scale_loss, scale_loss_dict, scale_info = self._compute_scale_loss(
                pred, tgt, scale_idx
            )
            scale_loss_dicts.append(scale_loss_dict)
            # Weight and accumulate
            weighted_scale_loss = scale_weight * scale_loss
            total_loss += weighted_scale_loss
            
            # Accumulate component losses
            for key in combined_losses.keys():
                if key in scale_loss_dict:
                    combined_losses[key] += scale_weight * scale_loss_dict[key]
            
            scale_stats.append(scale_info)
        
        # ⭐ STORE PER-BATCH SCALE DATA (unweighted raw losses)
        batch_scale_data = []
        for i, (stat, loss_dict) in enumerate(zip(scale_stats, scale_loss_dicts)):
            batch_scale_data.append({
                'scale': i,
                'grid': stat['grid_size'],
                'num_pos': stat['num_pos'],
                'obj': loss_dict['obj'],
                'cls': loss_dict['cls'],
                'iou': loss_dict['iou'],
                'center': loss_dict['center'],
                'dims': loss_dict['dims'],
                'rot': loss_dict['rot'],
                'iou_pred': loss_dict['iou_pred'],
                'total': loss_dict['total']
            })
        self.epoch_scale_stats.append(batch_scale_data)
        
        self.last_losses = combined_losses
        self.scale_stats = scale_stats
        return total_loss
    
    def _compute_scale_loss(self, pred, tgt, scale_idx):
        """
        MOVED: Original forward() logic, now processes single scale
        Returns: (loss, loss_dict, info_dict)
        """
        B, C, H, W = pred.shape
        
        obj_pred = pred[:, 0:1, ...]   # objectness logits
        box_raw  = pred[:, 1:8, ...]   # raw regression
        cls_pred = pred[:, 8:8+self.num_classes, ...]    # class logits
        iou_pred = pred[:, -1:, ...]    # IOU prediction logits

        obj_true = tgt['objectness'].to(pred.device)
        box_true = tgt['boxes'].to(pred.device)
        cls_true = tgt['classes'].to(pred.device)

        pos_mask = (obj_true == 1.0).squeeze(1)   # (B,H,W)
        heat_mask = (obj_true > 0.5).squeeze(1)    # (B, H, W)
        num_pos = pos_mask.sum().float()
        num_heat = heat_mask.sum().float()
        
        # NEW: BEV range varies by scale for proper grid mapping
        X_MIN, X_MAX = -50.0, 50.0
        Z_MIN, Z_MAX = 0.0, 100.0
        Y_MIN, Y_MAX = -5.0, 5.0

        # ========== OBJECTNESS LOSS ==========
        num_pos_safe = max(num_pos.item(), 1)
        obj_loss_sum = self.focal_loss(obj_pred.squeeze(1), obj_true.squeeze(1))
        obj_loss = obj_loss_sum / (num_pos_safe)
        # obj_loss = obj_loss_sum

        # ========== CLASSIFICATION LOSS ==========
        if num_pos > 0:
            cls_pred_pos = cls_pred.permute(0, 2, 3, 1)[pos_mask]
            # cls_true_pos = cls_true.permute(0, 2, 3, 1)[pos_mask]
            cls_true_pos = cls_true[pos_mask]
            cls_loss = F.cross_entropy(cls_pred_pos, cls_true_pos, reduction='none')
            weights = self.class_weights.to(cls_pred.device)[cls_true_pos.long()]
            # cls_loss = cls_loss_sum / num_pos
            # NEW: Apply class balancing weights
            cls_loss = (cls_loss * weights).mean()
            cls_loss = cls_loss.mean()
            if torch.isnan(cls_loss):
                print("class loss is NaN")
                cls_loss = torch.tensor(0.0, device=pred.device)
        else:
            cls_loss = torch.tensor(0.0, device=pred.device)

        # ========== BOX REGRESSION ==========
        if num_pos > 0:
            box_pred_pos_raw = box_raw.permute(0, 2, 3, 1)[pos_mask]
            box_true_pos     = box_true.permute(0, 2, 3, 1)[pos_mask]

            pos_idx = torch.nonzero(pos_mask, as_tuple=False)
            gh = pos_idx[:, 1].float()
            gw = pos_idx[:, 2].float()

            # Decode offsets
            x_off = torch.sigmoid(box_pred_pos_raw[:, 0])
            z_off = torch.sigmoid(box_pred_pos_raw[:, 2])

            # Decode offsets in raw logits so that neighbor cells can also detect objects\
            # x_off = box_pred_pos_raw[:, 0].clamp(min=0.0, max=1.0)
            # z_off = box_pred_pos_raw[:, 2].clamp(min=0.0, max=1.0)

            # Grid to metric conversion
            x_norm = (gw + x_off) / W
            z_norm = (gh + z_off) / H

            x_pred = X_MIN + x_norm * (X_MAX - X_MIN)
            z_pred = Z_MIN + z_norm * (Z_MAX - Z_MIN)

            # Y coordinate
            y_norm = torch.sigmoid(box_pred_pos_raw[:, 1])
            y_pred = Y_MIN + y_norm * (Y_MAX - Y_MIN)

            # Dimensions (log space)
            log_dims = box_pred_pos_raw[:, 3:6].clamp(min=-3.0, max=4.0)
            dims_m  = torch.exp(log_dims).clamp(min=0.1, max=100.0)
            w_pred, l_pred, h_pred = dims_m[:, 0], dims_m[:, 1], dims_m[:, 2]

            # Rotation
            yaw_pred = box_pred_pos_raw[:, 6]

            box_pred_metric = torch.stack(
                [x_pred, y_pred, z_pred, w_pred, l_pred, h_pred, yaw_pred],
                dim=1
            )
            box_true_metric = box_true_pos.clone()


            iou = iou_3d_rotated_metric(box_pred_metric[:, :7], box_true_metric[:, :7])

            # Center loss (metric space)
            center_pred = torch.stack([x_pred, y_pred, z_pred], dim=1)
            center_true = torch.stack([
                box_true_metric[:, 0],
                box_true_metric[:, 1],
                box_true_metric[:, 2]
            ], dim=1)
            center_loss = self.smooth_l1(center_pred, center_true)
            center_loss = center_loss.mean()
            if torch.isnan(center_loss):
                print("Center loss is NaN")


            # Dimensions loss (log space)
            eps = 1e-6
            dim_p = box_pred_metric[:, 3:6].clamp(min=eps)
            dim_t = box_true_metric[:, 3:6].clamp(min=eps)
            log_p = torch.log(dim_p)
            log_t = torch.log(dim_t)
            dims_loss = self.smooth_l1(log_p, log_t).mean()
            if torch.isnan(dims_loss):
                print("Dims loss is NaN")

            # Rotation loss
            rot_loss = rotation_loss(box_pred_metric[:, 6], box_true_metric[:, 6])
            if torch.isnan(rot_loss):
                print("Rot loss is NaN")
            
            # IoU loss
            
            iou_loss = (1.0 - iou)
            iou_loss = iou_loss.mean()
            if torch.isnan(iou_loss) or iou_loss > 2.0:
                iou_loss = torch.tensor(1.0, device=pred.device)
            
        else:
            iou_loss = torch.tensor(0.0, device=pred.device)
            center_loss = torch.tensor(0.0, device=pred.device)
            dims_loss = torch.tensor(0.0, device=pred.device)
            rot_loss = torch.tensor(0.0, device=pred.device)

        if num_heat>0:
            box_pred_pos_raw = box_raw.permute(0, 2, 3, 1)[heat_mask]
            box_true_pos     = box_true.permute(0, 2, 3, 1)[heat_mask]

            pos_idx = torch.nonzero(heat_mask, as_tuple=False)
            gh = pos_idx[:, 1].float()
            gw = pos_idx[:, 2].float()

            # # Decode offsets
            x_off = torch.sigmoid(box_pred_pos_raw[:, 0])
            z_off = torch.sigmoid(box_pred_pos_raw[:, 2])

            # Decode offsets in raw logits so that neighbor cells can also detect objects\
            # x_off = box_pred_pos_raw[:, 0]
            # z_off = box_pred_pos_raw[:, 2]

            # Grid to metric conversion
            x_norm = (gw + x_off) / W
            z_norm = (gh + z_off) / H

            x_pred = X_MIN + x_norm * (X_MAX - X_MIN)
            z_pred = Z_MIN + z_norm * (Z_MAX - Z_MIN)

            # Y coordinate
            y_norm = torch.sigmoid(box_pred_pos_raw[:, 1])
            y_pred = Y_MIN + y_norm * (Y_MAX - Y_MIN)

            # Dimensions (log space)
            log_dims = box_pred_pos_raw[:, 3:6].clamp(min=-3.0, max=4.0)
            dims_m  = torch.exp(log_dims).clamp(min=0.1, max=100.0)
            w_pred, l_pred, h_pred = dims_m[:, 0], dims_m[:, 1], dims_m[:, 2]

            # Rotation
            yaw_pred = box_pred_pos_raw[:, 6]

            box_pred_metric = torch.stack(
                [x_pred, y_pred, z_pred, w_pred, l_pred, h_pred, yaw_pred],
                dim=1
            )
            box_true_metric = box_true_pos.clone()

            iou = iou_3d_rotated_metric(box_pred_metric[:, :7], box_true_metric[:, :7])


            iou_pred_pos = iou_pred.squeeze(1)[heat_mask]    # (N_pos, ) logits
            iou_target = torch.min(
                torch.tensor(1.0, device=iou.device),
                torch.max(torch.tensor(0.0, device=iou.device), 2*iou-0.5)
            )

            #iou prediction loss
            iou_pred_loss = self.bce_loss(iou_pred_pos, iou_target).mean()
            
        else:
            iou_pred_loss = torch.tensor(0.0, device=pred.device)
        
        

        # ========== TOTAL SCALE LOSS ==========
        scale_loss = (
            self.iou_loss_weight * iou_loss
            + self.center_loss_weight * center_loss
            + self.dims_loss_weight * dims_loss
            + self.cls_loss_weight * cls_loss
            + self.obj_loss_weight * obj_loss
            + self.rot_loss_weight * rot_loss
            + self.iou_pred_weight * iou_pred_loss
        )

        # Return loss components for logging
        loss_dict = {
            'total': scale_loss.item(),
            'obj': obj_loss.item(),
            'cls': cls_loss.item(),
            'iou': iou_loss.item(),
            'center': center_loss.item(),
            'dims': dims_loss.item(),
            'rot': rot_loss.item(),
            'iou_pred': iou_pred_loss.item()
        }
        
        # NEW: Scale info for debugging
        info_dict = {
            'scale_idx': scale_idx,
            'grid_size': (H, W),
            'num_pos': num_pos.item(),
            'num_total': B * H * W,
        }

        return scale_loss, loss_dict, info_dict
    
    def get_epoch_averages(self):
        """Compute average losses across ALL batches in epoch_scale_stats"""
        if not self.epoch_scale_stats:
            return None
        
        # Average per scale across all batches
        scale_avgs = {0: [], 1: [], 2: []}
        
        for batch_data in self.epoch_scale_stats:
            for scale_data in batch_data:
                scale_idx = scale_data['scale']
                scale_avgs[scale_idx].append(scale_data)
        
        averages = []
        for scale_idx in [0, 1, 2]:
            scale_data = scale_avgs[scale_idx]
            if not scale_data:
                continue
                
            avg = {
                'scale': scale_idx,
                'grid': scale_data[0]['grid'],
                'num_pos_avg': np.mean([d['num_pos'] for d in scale_data]),
                'obj_avg': np.mean([d['obj'] for d in scale_data]),
                'cls_avg': np.mean([d['cls'] for d in scale_data]),
                'iou_avg': np.mean([d['iou'] for d in scale_data]),
                'center_avg': np.mean([d['center'] for d in scale_data]),
                'dims_avg': np.mean([d['dims'] for d in scale_data]),
                'rot_avg': np.mean([d['rot'] for d in scale_data]),
                'iou_pred_avg': np.mean([d['iou_pred'] for d in scale_data]),
                'total_avg': np.mean([d['total'] for d in scale_data])
            }
            averages.append(avg)
        
        self.scale_averages = averages
        return averages

