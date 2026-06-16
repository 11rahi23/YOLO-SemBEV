import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from hydra_backward_proj import HyDRaBackwardProjection
import torchvision.models as models

#Metric BEV configuration (camera frame: x right, y down, z forward)
X_MIN, X_MAX = -50.0, 50.0    # Lateral range in meters
Z_MIN, Z_MAX = 0.0, 100.0       # Depth range in meters
Y_MIN, Y_MAX = -10.0, 10.0      # Vertical range in meters

BEV_H, BEV_W = 128, 128        # z, x resolution of BEV
DEPTH_BINS = 128              # number of depth bins for image lifting


    
class SparseSemanticProposals(nn.Module):
    """Generate coarse proposals from semantic peaks"""

    def __init__(self, 
                 num_classes=12, 
                 foreground_classes=[2,6,7,8,9,10,11], 
                 confidence_threshold=0.5,
                 max_peaks_per_class=20,
                 peak_kernel_size=7
                 ):
        super().__init__()
        self.foreground_classes = foreground_classes
        self.confidence_threshold = confidence_threshold
        self.max_peaks_per_class = max_peaks_per_class
        self.peak_kernel_size = peak_kernel_size
        self.proposal_encoder = nn.Sequential(
            nn.Linear(3 + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )    # [x, y, z] + class_probs

    def _backproject_pixel(self, u, v, depth, camK):
        """Back-project a single pixel to 3D camera coordinates.
            Args:
                u: float, pixel column (x_coordinate)
                v: float, pixel row (y-coordinate)
                depth: float, metric depth in meters (z-coordinate)
                K: (3,3) camera intrinsic matrix
                
            Returns:
                xyz: (3, ) tensor [x, y, z] in camera frame (meters)"""
        
        device = camK.device

        # Extract intrinsic parameters
        fx = camK[0,0]
        fy = camK[1,1]
        cx = camK[0,2]
        cy = camK[1,2]

        # Back-project using pinhole camera model
        x = (u-cx)*depth / fx
        y = (v-cy)*depth / fy
        z = depth

        xyz = torch.stack([x,y,z])

        return xyz
        

    def forward(self, seg_mask, depth_prob, depth_values, cam_K, stride=8):
        """Instead of lifting all pixels,
            1. Find local maxima in segmentation (peak detection)
            2. Only lift ~200-200 peak pixels per image
            3. Generate proposal from peaks"""
        B, C, H, W = seg_mask.shape
        device = seg_mask.device
        _, D, H_depth, W_depth = depth_prob.shape

        if H != H_depth or W != W_depth:
            raise ValueError(f"Resolution mismatch! seg_mask ({H}x{W}) != depth_prob ({H_depth}x{W_depth})")

        # Ensure depth values in 1D
        if depth_values.dim() > 1:
            depth_values = depth_values.squeeze()
        depth_values = depth_values.to(device)

        proposals_batch = []
        xyz_batch = []

        for b in range (B):
            # Find local maxima for each foreground class
            proposals = []
            xyz_list = []
            for cls_index in self.foreground_classes:
                cls_mask = seg_mask[b, cls_index]   # (H, W)

                # Max pooling to find local peaks
                local_max = F.max_pool2d(
                    cls_mask.unsqueeze(0).unsqueeze(0),     # (1, 1, H ,W)
                    kernel_size=self.peak_kernel_size, stride=1, padding=self.peak_kernel_size //2
                ).squeeze()     # (H, W)

                # Only keep actual peaks above threshold
                is_peak = (cls_mask ==local_max.squeeze()) & (cls_mask>self.confidence_threshold)

                # Get peak location
                peak_coords = torch.nonzero(is_peak, as_tuple=False)    # (N,2)

                if len(peak_coords)==0:
                    continue

                # Limit to top_k peaks per class
                peak_scores = cls_mask[is_peak]
                topk = min(self.max_peaks_per_class, len(peak_coords))
                topk_scores, topk_idx = torch.topk(peak_scores, topk)

                peak_coords = peak_coords[topk_idx]     # (topk, 2)
            
                # Lift peaks to 3D
                for peak_coord in peak_coords:
                    v, u = peak_coord[0].item(), peak_coord[1].item()
                    # Sample depth at peak
                    depth_dist = depth_prob[b, :, v, u]
                    
                    expected_depth = (depth_dist * depth_values).sum()

                    # Cell center in original image space
                    u_orig = (u + 0.5) * stride
                    v_orig = (v + 0.5) * stride

                    # Back-project to 3D
                    xyz = self._backproject_pixel(u_orig, v_orig, expected_depth, cam_K[b])

                    # Encode proposal
                    cls_probs = seg_mask[b, :, v, u]
                    proposal_feat = self.proposal_encoder(torch.cat([xyz, cls_probs], dim=0))    # (64, )
                    xyz_list.append(xyz)
                    proposals.append(proposal_feat)     
                
            proposals_batch.append(proposals)
            xyz_batch.append(xyz_list)

        return proposals_batch, xyz_batch



class SparseProposalToBEV(nn.Module):
    """
    Encode sparse proposals (from SparseSemanticProposals) to BEV grid.
    Simpler than ProposalToBEVEncoder since we have fewer proposals.
    """
    
    def __init__(self, 
                 bev_h=128, 
                 bev_w=128,
                 proposal_dim=64,
                 bev_channels=64,
                 x_range=(-50.0, 50.0),
                 z_range=(0.0, 100.0),
                 gaussian_sigma=2.0):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.proposal_dim = proposal_dim
        self.bev_channels = bev_channels
        self.x_range = x_range
        self.z_range = z_range
        self.gaussian_sigma = gaussian_sigma
        
        # Project proposal features to BEV channels
        self.feature_proj = nn.Sequential(
            nn.Linear(proposal_dim, bev_channels),
            nn.ReLU(),
            nn.Linear(bev_channels, bev_channels)
        )
        
        # Refine BEV features with convolution
        self.bev_refiner = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(),
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU()
        )
    
    def forward(self, proposals_batch, xyz_batch, batch_size, device):
        bev_feat = torch.zeros(batch_size, self.bev_channels, 
                            self.bev_h, self.bev_w, device=device)
        
        # Precompute meshgrid ONCE
        i_coords = torch.arange(self.bev_h, dtype=torch.float32, device=device)
        j_coords = torch.arange(self.bev_w, dtype=torch.float32, device=device)
        ii, jj = torch.meshgrid(i_coords, j_coords, indexing='ij')  # (H, W)
        
        for b in range(batch_size):
            proposals = proposals_batch[b]
            xyz_list = xyz_batch[b]
            if len(proposals) == 0:
                continue
            
            proposals_tensor = torch.stack(proposals)       # (N, 64)
            bev_features = self.feature_proj(proposals_tensor)  # (N, C)
            
            # Vectorize over proposals too
            xyzs = torch.stack(xyz_list)  # (N, 3)
            x_norm = (xyzs[:, 0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
            z_norm = (xyzs[:, 2] - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
            
            valid = (x_norm >= 0) & (x_norm <= 1) & (z_norm >= 0) & (z_norm <= 1)
            if not valid.any():
                continue
            
            grid_j = (x_norm[valid] * self.bev_w).clamp(0, self.bev_w-1)
            grid_i = (z_norm[valid] * self.bev_h).clamp(0, self.bev_h-1)
            feats_v = bev_features[valid]  # (N_valid, C)
            
            # Vectorized gaussian: (N_valid, H, W)
            di = (ii.unsqueeze(0) - grid_i.view(-1, 1, 1)) ** 2
            dj = (jj.unsqueeze(0) - grid_j.view(-1, 1, 1)) ** 2
            gaussians = torch.exp(-(di + dj) / (2 * self.gaussian_sigma**2))
            
            # (N_valid, C, 1, 1) * (N_valid, 1, H, W) -> sum over N
            bev_feat[b] += (feats_v.view(-1, self.bev_channels, 1, 1) * 
                            gaussians.unsqueeze(1)).sum(0)
        
        return self.bev_refiner(bev_feat)
    def forward_depr(self, proposals_batch, xyz_batch, batch_size, device):
        """
        Args:
            proposals_batch: List[List[Tensor]] - proposals[b] = [(64,), (64,), ...]
            xyz_batch: List[List[Tensor]] - xyz_batch[b] = [(3,), (3,), ...]
            batch_size: int
            device: torch device
        
        Returns:
            bev_feat: (B, bev_channels, bev_h, bev_w)
        """
        bev_feat = torch.zeros(
            batch_size, self.bev_channels, 
            self.bev_h, self.bev_w, device=device
        )
        
        for b in range(batch_size):
            proposals = proposals_batch[b]
            xyz_list = xyz_batch[b]
            
            if len(proposals) == 0:
                continue
            
            # Stack proposals: (N, 64)
            proposals_tensor = torch.stack(proposals)
            
            # Project to BEV feature dimension
            bev_features = self.feature_proj(proposals_tensor)  # (N, bev_channels)
            
            for i, xyz in enumerate(xyz_list):
                x, y, z = xyz[0].item(), xyz[1].item(), xyz[2].item()
                
                # Convert to BEV grid coordinates
                x_norm = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
                z_norm = (z - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
                
                # Check bounds
                if not (0 <= x_norm <= 1 and 0 <= z_norm <= 1):
                    continue
                
                # Grid indices
                grid_j = int(x_norm * self.bev_w)
                grid_i = int(z_norm * self.bev_h)
                
                # Clamp to valid range
                grid_j = max(0, min(self.bev_w - 1, grid_j))
                grid_i = max(0, min(self.bev_h - 1, grid_i))
                
                # Create Gaussian heatmap centered at this proposal
                i_coords = torch.arange(self.bev_h, dtype=torch.float32, device=device)
                j_coords = torch.arange(self.bev_w, dtype=torch.float32, device=device)
                ii, jj = torch.meshgrid(i_coords, j_coords, indexing='ij')
                
                gaussian = torch.exp(
                    -((ii - grid_i)**2 + (jj - grid_j)**2) / (2 * self.gaussian_sigma**2)
                )  # (H, W)
                
                # Spread features weighted by Gaussian
                bev_feat[b] += bev_features[i].view(-1, 1, 1) * gaussian.unsqueeze(0)
        
        # Refine with convolution
        bev_feat = self.bev_refiner(bev_feat)
        
        return bev_feat



class RadarSemanticPainter(nn.Module):
    """Paint radar points with semantic segmentation scores"""
    def __init__(self):
        super().__init__()
    
    def forward(self, radar_points, seg_mask, cam_K):
        """
        radar_points: (B, N, 7) [x, y, z, vx, vy, cid, pop] in camera frame
        seg_mask: (B, num_classes, H, W) segmentation probabilities
        cam_K: (B, 3, 3) camera intrinsics
        
        Returns: (B, N, 7+num_classes) painted radar points
        """
        B, N, _ = radar_points.shape
        _, C, H, W = seg_mask.shape
        device = radar_points.device

        painted_features = []

        for b in range (B):
            # Project radar points to image
            pts = radar_points[b]   # (N, 7)
            xyz = pts[:, :3]    # (N, 3)
            
            # Valid points (depth>0)
            valid_mask = xyz[:,2]>0

            # Project to pixels
            K = cam_K[b]    # (3, 3)
            xyz_homo = torch.cat([xyz, torch.ones(N, 1, device=device)], dim=1) # (N, 4)
            pixels = K @ xyz_homo[:, :3].t()    # (3, N)
            pixels = pixels[:2] / (pixels[2:3] + 1e-6)  # (2, N)

            u = pixels[0]   # (N, )
            v = pixels[1]   # (N, )

            # Check bounds 
            in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_mask

            # Sample segmentation scores
            seg_features = torch.zeros(N, C, device=device)
            if in_bounds.any():
                u_valid = u[in_bounds].long()
                v_valid = v[in_bounds].long()
                seg_features[in_bounds] = seg_mask[b, :, v_valid, u_valid].t()  # (N_valid, C)
            
            # Concatenate: [x, y, z, vx, vy, cid, pop] + [seg_class_0, ....., seg_class_C]
            painted = torch.cat([pts, seg_features], dim=1) # (N, 7+C)
            painted_features.append(painted)
        
        return torch.stack(painted_features, dim=0)     # (B, N, 7+C)

class ImageDepthHead(nn.Module):
    """
    Predict per-pixel discrete depth probabilities and per-pixel features from an image feature map (e.g., P3 with stride 8)
    """
    def __init__(self, in_ch, num_depth_bins=DEPTH_BINS, feat_ch=64):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.intrinsic_encoder = nn.Sequential(
            nn.Linear(4, 64),   # fx, fy, cx, cy
            nn.ReLU(),
            nn.Linear(64, in_ch)
        )
        self.conv_feat = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.SiLU(),
        )
        self.conv_depth = nn.Conv2d(feat_ch, num_depth_bins, 1)

        #Metric depth bin centers in meters
        self.register_buffer(
            "depth_values",
            torch.linspace(Z_MIN, Z_MAX, num_depth_bins).view(1, num_depth_bins, 1, 1), # (1, D, 1, 1)
            persistent=False
        )

    def forward(self, x, cam_K):
        """
        # x: (B, C, H, W) image features  (e.g., P3)
        Returns: 
            feat: (B, F, H, W)
            depth_prob: (B, D, H, W)
            depth_value: (1, D, 1, 1) in meters
        """
        B, C, H, W = x.shape
        intrinsics = torch.stack([
            cam_K[:, 0, 0], cam_K[:, 1, 1],  # fx, fy
            cam_K[:, 0, 2], cam_K[:, 1, 2]   # cx, cy
        ], dim=-1)  # (B, 4)
        
        # Encode and broadcast
        cam_feat = self.intrinsic_encoder(intrinsics)  # (B, C)
        cam_feat = cam_feat.view(B, C, 1, 1).expand_as(x)
        
        x = x + cam_feat   # Inject camera geometry
        feat = self.conv_feat(x)    # (B, F, H, W)
        depth_logits = self.conv_depth(feat)   # (B, D, H, W)
        depth_prob = depth_logits.softmax(dim=1)    # (B, D, H, W)
        return feat, depth_prob, self.depth_values
    
class ImageBEVEncoder(nn.Module):
    """
    Lift image features + depth probabilities to a BEV feature map. Uses a simplified BEVDepth-style view transformer.
    """
    def __init__(
            self,
            feat_ch=64, 
            num_depth_bins=DEPTH_BINS,
            bev_h=BEV_H,
            bev_w=BEV_W,
            x_range=(X_MIN, X_MAX),
            z_range=(Z_MIN, Z_MAX),
            num_seg_classes=11
    ):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.feat_ch = feat_ch
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_range = x_range
        self.z_range = z_range
        self.num_seg_classes = num_seg_classes
    
    def forward(self, feat, depth_prob, depth_values, cam_K, seg_mask=None):
        """
        Vectorized version of BEV projection.
        feat: (B, F, H, W)
        depth_prob: (B, D, H, W)
        depth_values: (1, D, 1, 1) depth bin centers in meters
        cam_K: (B, 3, 3) camera intrinsics
        seg_mask: (B, num_classes, H_img, W_img)
        Returns: bev_feat: (B, F+num_seg_classes, H_bev, W_bev)
        """
        B, F, H, W = feat.shape
        D = depth_prob.size(1)
        device = feat.device

        # 1. Basic pixel grid (u,v)
        u = torch.linspace(0, W-1, W, device=device)
        v = torch.linspace(0, H-1, H, device=device)
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # (H,W)
        uv1 = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)  # (H, W, 3)
        uv1 = uv1.view(1, 1, H, W, 3).expand(B, D, H, W, 3)  # (B, D, H, W, 3)

        # 2. Depth values for each bin (center of bin)
        depth_vals = depth_values.to(device).view(1, D, 1, 1, 1)  # (1, D, 1, 1, 1)

        # 3. Back-project to camera frame: X_cam = depth * K^(-1) [u, v, 1]^T
        K_inv = torch.inverse(cam_K).view(B, 1, 1, 1, 3, 3)  # (B, 1, 1, 1, 3, 3)
        xyz_cam = torch.matmul(K_inv, uv1.unsqueeze(-1)).squeeze(-1)  # (B, D, H, W, 3)
        xyz_cam = xyz_cam * depth_vals  # (B, D, H, W, 3)

        # Extract x, z coordinates
        x = xyz_cam[..., 0]  # (B, D, H, W)
        z = xyz_cam[..., 2]  # (B, D, H, W)

        # 4. Map metric x,z to BEV grid indices
        x_norm = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        z_norm = (z - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        i = (z_norm * self.bev_h).long().clamp(0, self.bev_h - 1)  # (B, D, H, W)
        j = (x_norm * self.bev_w).long().clamp(0, self.bev_w - 1)  # (B, D, H, W)

        # 5. Lift image features with depth_prob
        feat_expanded = feat.unsqueeze(1).expand(B, D, F, H, W)  # (B, D, F, H, W)
        depth_prob_expanded = depth_prob.unsqueeze(2)  # (B, D, 1, H, W)
        lifted = feat_expanded * depth_prob_expanded  # (B, D, F, H, W)

        # 6. Vectorized accumulation into BEV
        # Flatten spatial dimensions
        lifted_flat = lifted.reshape(B, D * H * W, F)  # (B, D*H*W, F)
        i_flat = i.reshape(B, D * H * W)  # (B, D*H*W)
        j_flat = j.reshape(B, D * H * W)  # (B, D*H*W)
        flat_idx = i_flat * self.bev_w + j_flat  # (B, D*H*W)

        # Create BEV feature map using scatter_add
        bev_feat = torch.zeros(B, self.bev_h * self.bev_w, F, device=device)
        bev_feat = bev_feat.scatter_add_(1, flat_idx.unsqueeze(-1).expand(-1, -1, F), lifted_flat)

        # Compute counts for averaging
        counts = torch.zeros(B, self.bev_h * self.bev_w, device=device)
        ones = torch.ones(B, D * H * W, device=device)
        counts = counts.scatter_add_(1, flat_idx, ones)

        log_counts = torch.log1p(counts).unsqueeze(-1)

        # Average by counts
        # bev_feat = bev_feat / counts.clamp(min=1).unsqueeze(-1)
        bev_feat = bev_feat / (log_counts + 1e-6)
        bev_feat = bev_feat.view(B, self.bev_h, self.bev_w, F).permute(0, 3, 1, 2)  # (B, F, H_bev, W_bev)

        # Handle semantic segmentation if provided
        if seg_mask is not None:
            _, C_seg, H_seg, W_seg = seg_mask.shape

            # Resize seg_mask to match feat resolution if needed
            if H_seg != H or W_seg != W:
                seg_mask_resized = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)
            else:
                seg_mask_resized = seg_mask

            # Lift semantic features with depth
            seg_expanded = seg_mask_resized.unsqueeze(1).expand(B, D, C_seg, H, W)  # (B, D, C_seg, H, W)
            seg_lifted = seg_expanded * depth_prob_expanded  # (B, D, C_seg, H, W)

            # Flatten and accumulate
            seg_lifted_flat = seg_lifted.reshape(B, D * H * W, C_seg)  # (B, D*H*W, C_seg)
            bev_seg = torch.zeros(B, self.bev_h * self.bev_w, C_seg, device=device)
            bev_seg = bev_seg.scatter_add_(1, flat_idx.unsqueeze(-1).expand(-1, -1, C_seg), seg_lifted_flat)

            # Average by counts (reuse counts from before)
            # bev_seg = bev_seg / counts.clamp(min=1).unsqueeze(-1)
            bev_seg = bev_seg / (log_counts + 1e-6)
            bev_seg = bev_seg.view(B, self.bev_h, self.bev_w, C_seg).permute(0, 3, 1, 2)  # (B, C_seg, H_bev, W_bev)

            # Concatenate semantic BEV with feature BEV
            bev_feat = torch.cat([bev_feat, bev_seg], dim=1)  # (B, F+C_seg, H_bev, W_bev)

        return bev_feat

    def _forward_non_vec(self, feat, depth_prob, depth_values, cam_K, seg_mask=None):
        """
        feat: (B, F, H, W)
        depth_prob: (B, D, H, W)
        depth_values: (1, D, 1, 1) depth bin centers in meters
        cam_K: (B, 3, 3) camera intrinsics
        cam_T: (B, 4, 4)    # camera->camera-ego? Not required
        seg_mask: (B, num_classes, H_img, W_img)
        Returns: bev_feat: (B, F+num_seg_classes, H_bev, W_bev)
        """
        B, F, H, W = feat.shape
        D = depth_prob.size(1)
        device = feat.device

        # 1. Basic pixel grid (u,v)
        u = torch.linspace(0, W-1, W, device=device)
        v = torch.linspace(0, H-1, H, device=device)
        vv, uu = torch.meshgrid(v,u, indexing='ij') # (H,W)
        uv1 = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)    # (H, W, 3)

        uv1 = uv1.view(1, 1, H, W, 3).repeat(B, D, 1, 1, 1) # (B, D, H, W, 3)

        # 2. Depth values for each bin (center of bin)
        depth_vals = depth_values.to(device)        # (1, D, 1, 1)
        depth_vals = depth_vals.view(1, D, 1, 1, 1) # (1, D, 1, 1, 1)

        # 3. Back-project to camera frame: X_cam = depth * K^(-1) [u, v, 1]^T
        K_inv = torch.inverse(cam_K).view(B, 1, 1, 1, 3, 3)     # (B, 1, 1, 1, 3, 3)
        uv1_col = uv1.unsqueeze(-1)                             # (B, D, H, W, 3, 1)
        xyz_cam = torch.matmul(K_inv, uv1_col).squeeze(-1)   # (B, D, H, W, 3)
        xyz_cam = xyz_cam*depth_vals            # (B, D, H, W, 3)

        # If cam_T is camera->ego and I want ego coords:
        

        x = xyz_cam[...,0]  # (B, D, H, W)
        z = xyz_cam[..., 2] # (B, D, H, W)

        # 4. Map metric x,z to BEV grid indices
        x_norm = (x-self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        z_norm = (z-self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        i = (z_norm * self.bev_h).long().clamp(0, self.bev_h - 1)   # row (z)
        j = (x_norm * self.bev_w).long().clamp(0, self.bev_w -1)    # col (x)

        # 5. Lift image features with depth_prob
        feat = feat.view(B, 1, F, H, W).repeat(1, D, 1, 1, 1)   # (B, D, F, H, W)
        depth_prob = depth_prob.unsqueeze(2)                    # (B, D, 1, H, W)
        lifted = feat * depth_prob                              # (B, D, F, H, W)

        bev_feat = torch.zeros(B, F, self.bev_h, self.bev_w, device=device)

        # 6. Accumualate into BEV (per batch, for clarity; can be vectorized later)
        for b in range (B):
            idx_i = i[b].reshape(-1)
            idx_j = j[b].reshape(-1)
            feats_b = lifted[b].permute(1, 0, 2, 3).reshape(F, -1).t()  # (D*H*W, F)
            flat_idx = idx_i * self.bev_w + idx_j                       # (D*H*W)
            bev_flat = torch.zeros(self.bev_h * self.bev_w, F, device=device)
            bev_flat.index_add_(0, flat_idx, feats_b)

            counts = torch.zeros(self.bev_h * self.bev_w, device=device)
            counts.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))
            bev_flat = bev_flat / counts.clamp(min=1).unsqueeze(-1)

            bev_feat[b] = bev_flat.view(self.bev_h, self.bev_w, F).permute(2,0,1)
        
        if seg_mask is not None:
            _, C_seg, H_seg, W_seg = seg_mask.shape

            # Resize seg_mask to match feat resolution if needed
            if H_seg != H or W_seg !=W:
                seg_mask_resized = F.interpolate(seg_mask, size=(H,W), mode='bilinear')
            else:
                seg_mask_resized = seg_mask
            
            # Lift semantic features with depth
            seg_lifted = seg_mask_resized.view(B, 1, C_seg, H, W).repeat(1, D, 1, 1, 1)     # (B, D, C, H, W)
            #depth_prob_exp = depth_prob.unsqueeze(2)    # (B, D, 1, H, W)
            seg_lifted = seg_lifted*depth_prob      # (B, D, C, H, W)
            bev_seg = torch.zeros(B, C_seg, self.bev_h, self.bev_w, device=device)

            for b in range(B):
                idx_i = i[b].reshape(-1)
                idx_j = j[b].reshape(-1)
                seg_b = seg_lifted[b].permute(1, 0, 2, 3).reshape(C_seg, -1).t()    # (D*H*W, C_seg)
                flat_idx = idx_i * self.bev_w + idx_j
                seg_flat = torch.zeros(self.bev_h*self.bev_w, C_seg, device=device)
                seg_flat.index_add_(0, flat_idx, seg_b)

                counts = torch.zeros(self.bev_h * self.bev_w, device=device)
                counts.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))
                seg_flat = seg_flat / counts.clamp(min=1).unsqueeze(-1)
                bev_seg[b] = seg_flat.view(self.bev_h, self.bev_w, C_seg).permute(2,0,1)
            
            # Concatenate semantic BEV with feature BEV
            bev_feat = torch.cat([bev_feat, bev_seg], dim=1)    # (B, F+C_seg, H_bev, W_bev)
        
        return bev_feat

class ImageBEVEncoderHeavy(nn.Module):
    """
    CUDA-optimized encoder that lifts image features to BEV using depth distribution.
    Fully vectorized - no batch loops.
    """
    def __init__(self, feat_ch=256, num_depth_bins=64, 
                 bev_h=128, bev_w=128,
                 x_range=(X_MIN, X_MAX), z_range=(Z_MIN, Z_MAX),
                 num_seg_classes=12):
        super().__init__()
        self.feat_ch = feat_ch
        self.num_depth_bins = num_depth_bins
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_range = x_range
        self.z_range = z_range
        self.num_seg_classes = num_seg_classes
        
    def forward(self, img_feat, depth_prob, depth_values, cam_K, seg_mask=None):
        """
        CUDA-optimized forward pass - fully vectorized.
        
        Args:
            img_feat: (B, F, H, W) - image features
            depth_prob: (B, D, H, W) - depth probability distribution
            depth_values: (D,) - depth bin centers in meters
            cam_K: (B, 3, 3) - camera intrinsics
            seg_mask: (B, C_seg, H, W) - semantic segmentation (optional)
        
        Returns:
            bev_feat: (B, F+C_seg, bev_h, bev_w) - BEV feature map
        """
        B, F, H, W = img_feat.shape
        D = depth_prob.shape[1]
        device = img_feat.device
        
        # ============ 1. Create pixel grid (vectorized) ============
        # u, v coordinates for each pixel
        u = torch.linspace(0, W-1, W, device=device)
        v = torch.linspace(0, H-1, H, device=device)
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # (H, W)
        
        # Homogeneous coordinates: (H, W, 3)
        uv1 = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
        
        # Expand for batch and depth: (B, D, H, W, 3)
        uv1 = uv1.unsqueeze(0).unsqueeze(0).expand(B, D, H, W, 3)
        
        # ============ 2. Depth values for each bin ============
        depth_vals = depth_values.to(device)  # (D,)
        depth_vals = depth_vals.view(1, D, 1, 1, 1)  # (1, D, 1, 1, 1)
        
        # ============ 3. Back-project to camera frame (vectorized) ============
        # K_inv: (B, 1, 1, 1, 3, 3)
        K_inv = torch.inverse(cam_K).view(B, 1, 1, 1, 3, 3)
        
        # Back-project: X_cam = depth * K^-1 @ [u, v, 1]^T
        uv1_col = uv1.unsqueeze(-1)  # (B, D, H, W, 3, 1)
        xyz_cam = torch.matmul(K_inv, uv1_col).squeeze(-1)  # (B, D, H, W, 3)
        xyz_cam = xyz_cam * depth_vals  # (B, D, H, W, 3)
        
        # Extract x, z coordinates (lateral and depth)
        x = xyz_cam[..., 0]  # (B, D, H, W)
        z = xyz_cam[..., 2]  # (B, D, H, W)
        
        # ============ 4. Map to BEV grid indices (vectorized) ============
        x_norm = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        z_norm = (z - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        
        i = (z_norm * self.bev_h).long().clamp(0, self.bev_h - 1)  # (B, D, H, W) - row (z)
        j = (x_norm * self.bev_w).long().clamp(0, self.bev_w - 1)  # (B, D, H, W) - col (x)
        
        # ============ 5. Lift image features with depth (vectorized) ============
        # Expand features for depth dimension: (B, D, F, H, W)
        feat = img_feat.unsqueeze(1).expand(B, D, F, H, W)
        
        # Weight by depth probability: (B, D, 1, H, W)
        depth_prob_exp = depth_prob.unsqueeze(2)
        
        # Weighted features: (B, D, F, H, W)
        lifted = feat * depth_prob_exp
        
        # ============ 6. Accumulate into BEV (fully vectorized scatter) ============
        # Flatten all dimensions for scatter
        # Create batch indices: (B, D, H, W)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, D, H, W)
        
        # Flatten spatial indices
        i_flat = i.reshape(-1)  # (B*D*H*W,)
        j_flat = j.reshape(-1)  # (B*D*H*W,)
        batch_flat = batch_idx.reshape(-1)  # (B*D*H*W,)
        
        # Features to scatter: (B*D*H*W, F)
        feats_flat = lifted.permute(0, 1, 3, 4, 2).reshape(-1, F)  # (B*D*H*W, F)
        
        # Compute global flattened index: batch * bev_h * bev_w + i * bev_w + j
        flat_idx = batch_flat * (self.bev_h * self.bev_w) + i_flat * self.bev_w + j_flat
        
        # Initialize BEV tensor
        bev_feat_flat = torch.zeros(B * self.bev_h * self.bev_w, F, device=device)
        counts_flat = torch.zeros(B * self.bev_h * self.bev_w, device=device)
        
        # Scatter add (single CUDA kernel for all batches, depths, pixels)
        bev_feat_flat.index_add_(0, flat_idx, feats_flat)
        counts_flat.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        
        # Average by count
        bev_feat_flat = bev_feat_flat / counts_flat.clamp(min=1).unsqueeze(-1)
        
        # Reshape to (B, bev_h, bev_w, F) -> (B, F, bev_h, bev_w)
        bev_feat = bev_feat_flat.view(B, self.bev_h, self.bev_w, F).permute(0, 3, 1, 2)
        
        # ============ 7. Lift semantic features (if provided) ============
        if seg_mask is not None:
            _, C_seg, H_seg, W_seg = seg_mask.shape
            
            # Resize seg_mask to match feature resolution if needed
            if H_seg != H or W_seg != W:
                seg_mask_resized = F.interpolate(seg_mask, size=(H, W), mode='bilinear')
            else:
                seg_mask_resized = seg_mask
            
            # Lift segmentation with depth: (B, D, C_seg, H, W)
            seg_lifted = seg_mask_resized.unsqueeze(1).expand(B, D, C_seg, H, W)
            seg_lifted = seg_lifted * depth_prob_exp  # Weight by depth
            
            # Flatten for scatter: (B*D*H*W, C_seg)
            seg_flat = seg_lifted.permute(0, 1, 3, 4, 2).reshape(-1, C_seg)
            
            # Initialize semantic BEV
            bev_seg_flat = torch.zeros(B * self.bev_h * self.bev_w, C_seg, device=device)
            
            # Scatter add (reuse same flat_idx and counts)
            bev_seg_flat.index_add_(0, flat_idx, seg_flat)
            bev_seg_flat = bev_seg_flat / counts_flat.clamp(min=1).unsqueeze(-1)
            
            # Reshape to (B, C_seg, bev_h, bev_w)
            bev_seg = bev_seg_flat.view(B, self.bev_h, self.bev_w, C_seg).permute(0, 3, 1, 2)
            
            # Concatenate semantic with features
            bev_feat = torch.cat([bev_feat, bev_seg], dim=1)  # (B, F+C_seg, bev_h, bev_w)
        
        return bev_feat
 



class RadarFeatureEncoder(nn.Module):
    """
    Encode radar points into BEV features at multiple channel widths.
    """
    def __init__(self,
                 grid_size=(BEV_H, BEV_W),
                 input_dim=7+12,                 # [x,y,z,vx,vy,cid,pop]
                 out_channels=(256, 512, 1024),
                 x_range=(X_MIN, X_MAX),
                 z_range=(Z_MIN, Z_MAX)):
        super().__init__()
        self.grid_size  = grid_size
        self.input_dim  = input_dim
        self.x_range    = x_range
        self.z_range    = z_range
        self.out_channels = out_channels

        self.linears = nn.ModuleList([nn.Linear(input_dim, ch)
                                      for ch in out_channels])

    def forward(self, radar_points):
        """
        radar_points: (B, N, 7) in CAMERA/EGO FRAME [x, y, z, vx, vy, cid, pop]
        Returns:
            bev_feats: list of BEV grids: [(B, C1, H, W), (B, C2, H, W), ...]
        """
        B, N, D = radar_points.shape
        H, W    = self.grid_size
        device  = radar_points.device

        x_cam = radar_points[..., 0]   # lateral
        z_cam = radar_points[..., 2]   # depth
        valid_mask = z_cam > 0         # (B, N)

        # normalize to [0,1] for indexing
        x_norm = (x_cam - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        z_norm = (z_cam - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        i_all = (z_norm * H).long().clamp(0, H - 1)   # rows
        j_all = (x_norm * W).long().clamp(0, W - 1)   # cols

        bev_feats = []

        for linear, out_dim in zip(self.linears, self.out_channels):
            point_feats = linear(radar_points)        # (B, N, out_dim)
            bev_feat = torch.zeros(B, H, W, out_dim, device=device)

            for b in range(B):
                valid_b = valid_mask[b]              # (N,)
                if valid_b.sum() == 0:
                    continue

                i_b = i_all[b, valid_b]              # (N_valid,)
                j_b = j_all[b, valid_b]
                feats_b = point_feats[b, valid_b]    # (N_valid, C)

                flat_idx = i_b * W + j_b             # (N_valid,)
                feat_flat = torch.zeros(H * W, out_dim, device=device)
                feat_flat.index_add_(0, flat_idx, feats_b)

                counts = torch.zeros(H * W, device=device)
                counts.index_add_(0, flat_idx,
                                  torch.ones_like(flat_idx, dtype=torch.float))

                feat_flat = feat_flat / counts.clamp(min=1).unsqueeze(-1)
                bev_feat[b] = feat_flat.view(H, W, out_dim)

            bev_feats.append(bev_feat.permute(0, 3, 1, 2))  # (B,C,H,W)

        return bev_feats
    
class RadarFeatureEncoderMultiScale(nn.Module):
    """
    CUDA-optimized encoder for radar points to multi-scale BEV features.
    Fully vectorized - no batch loops.
    """
    def __init__(self,
                 grid_sizes=[(128, 128), (64, 64), (32, 32)],
                 input_dim=7+12,
                 out_channels=(256, 512, 1024),
                 x_range=(X_MIN, X_MAX),
                 z_range=(Z_MIN, Z_MAX)):
        super().__init__()
        self.grid_sizes = grid_sizes
        self.input_dim = input_dim
        self.x_range = x_range
        self.z_range = z_range
        self.out_channels = out_channels
        
        # Single linear projection per scale
        self.linears = nn.ModuleList([
            nn.Linear(input_dim, ch) for ch in out_channels
        ])
        
    def forward(self, radar_points):
        """
        CUDA-optimized forward pass - fully vectorized.
        
        Args:
            radar_points: (B, N, input_dim)
        
        Returns:
            bev_feats: List of [(B, C1, H1, W1), (B, C2, H2, W2), ...]
        """
        B, N, D = radar_points.shape
        device = radar_points.device
        
        # Extract and normalize coordinates (vectorized for all batches)
        x_cam = radar_points[:, :, 0]  # (B, N)
        z_cam = radar_points[:, :, 2]  # (B, N)
        valid_mask = z_cam > 0         # (B, N)
        
        # Normalize to [0, 1]
        x_norm = (x_cam - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        z_norm = (z_cam - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        
        bev_feats = []
        
        for scale_idx, (grid_size, out_dim) in enumerate(zip(self.grid_sizes, self.out_channels)):
            H, W = grid_size
            
            # Compute grid indices (vectorized)
            i_grid = (z_norm * H).long().clamp(0, H - 1)  # (B, N)
            j_grid = (x_norm * W).long().clamp(0, W - 1)  # (B, N)
            
            # Project features (vectorized for entire batch)
            point_feats = self.linears[scale_idx](radar_points)  # (B, N, out_dim)
            
            # Flatten batch and spatial dimensions for scatter
            # Create batch indices
            batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)  # (B, N)
            
            # Flatten all dimensions
            batch_flat = batch_idx.reshape(-1)       # (B*N,)
            i_flat = i_grid.reshape(-1)              # (B*N,)
            j_flat = j_grid.reshape(-1)              # (B*N,)
            valid_flat = valid_mask.reshape(-1)      # (B*N,)
            feats_flat = point_feats.reshape(-1, out_dim)  # (B*N, out_dim)
            
            # Filter invalid points
            valid_indices = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
            if valid_indices.numel() == 0:
                # No valid points - return zeros
                bev_feats.append(torch.zeros(B, out_dim, H, W, device=device))
                continue
            
            batch_valid = batch_flat[valid_indices]  # (N_valid,)
            i_valid = i_flat[valid_indices]          # (N_valid,)
            j_valid = j_flat[valid_indices]          # (N_valid,)
            feats_valid = feats_flat[valid_indices]  # (N_valid, out_dim)
            
            # Compute flattened indices: batch*H*W + row*W + col
            flat_idx = batch_valid * (H * W) + i_valid * W + j_valid  # (N_valid,)
            
            # Initialize output tensor
            bev_feat_flat = torch.zeros(B * H * W, out_dim, device=device)
            counts_flat = torch.zeros(B * H * W, device=device)
            
            # Scatter add (single CUDA kernel launch for all batches)
            bev_feat_flat.index_add_(0, flat_idx, feats_valid)
            counts_flat.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
            
            # Average by count (vectorized division)
            bev_feat_flat = bev_feat_flat / counts_flat.clamp(min=1).unsqueeze(-1)
            
            # Reshape to (B, H, W, out_dim) then permute to (B, out_dim, H, W)
            bev_feat = bev_feat_flat.view(B, H, W, out_dim).permute(0, 3, 1, 2)
            bev_feats.append(bev_feat)
        
        return bev_feats


    
   
class ConvBNAct(nn.Module):
    """Basic Conv-Batchnorm-Activation block"""
    def __init__(self, in_c, out_c, k=3, s=1, p=None, act = True):
        super().__init__()
        if p is None: 
            p = k//2
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class C2f(nn.Module):
    """Efficient Cross-Stage Partial Module"""
    def __init__(self, in_c, out_c, n=2):
        super().__init__()
        c_ = out_c //2
        self.cv1 = ConvBNAct(in_c, c_, k=1, s=1)
        self.cv2 = ConvBNAct(in_c, c_, k=1, s=1)
        self.blocks = nn.Sequential(*[ConvBNAct(c_, c_, 3) for _ in range(n)])
        self.cv3 = ConvBNAct(c_*(n+2), out_c, k=1, s=1)
    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        outputs = [y1,y2]
        for block in self.blocks:
            y2 = block(y2)
            outputs.append(y2)
        out = torch.cat(outputs, dim = 1)
        return self.cv3(out)
    
class BEVFusionBlock(nn.Module):
    def __init__(self, img_ch, radar_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(img_ch + radar_ch, out_ch, 3, 1),
            C2f(out_ch, out_ch, n=2)
        )
    
    def forward(self, img_bev, radar_bev):
        x = torch.cat([img_bev, radar_bev], dim=1)
        return self.conv(x)

# ================= Hydra Radar Guided Depth Consistency ===================================

class SEFusionBlock(nn.Module):
    """
    Squeeze-and-Excitation fusion of image BEV and radar BEV.
    Equivalent to HyDRa step 2: 'splatted semantic BEV features and 
    radar-BEV features are concatenated and fused by a SE block'
    """
    def __init__(self, img_ch, radar_ch, out_ch, reduction=16):
        super().__init__()
        combined = img_ch + radar_ch

        # Channel mixing
        self.mix = nn.Sequential(
            nn.Conv2d(combined, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        # SE: global context → channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, out_ch, 1, 1)
            nn.Flatten(),                      # (B, out_ch)
            nn.Linear(out_ch, out_ch // reduction),
            nn.ReLU(),
            nn.Linear(out_ch // reduction, out_ch),
            nn.Sigmoid()
        )

    def forward(self, img_bev, radar_bev):
        x = torch.cat([img_bev, radar_bev], dim=1)   # (B, img_ch+radar_ch, H, W)
        x = self.mix(x)                               # (B, out_ch, H, W)
        scale = self.se(x).view(x.shape[0], -1, 1, 1) # (B, out_ch, 1, 1)
        return x * scale                              # channel-wise recalibration
    

class RadarGuidanceNetwork(nn.Module):
    """
    HyDRa RGN: lightweight radar-only BEV → spatial attention weights.
    'A 3x3 convolution followed by sigmoid encodes radar-only BEV features 
    to additional attention weights'
    """
    def __init__(self, radar_ch):
        super().__init__()
        self.rgn = nn.Sequential(
            nn.Conv2d(radar_ch, radar_ch, 3, padding=1),
            nn.BatchNorm2d(radar_ch),
            nn.ReLU(),
            nn.Conv2d(radar_ch, 1, 3, padding=1),  # collapse to single attention map
            nn.Sigmoid()                            # (B, 1, H, W) weights in [0,1]
        )

    def forward(self, radar_bev):
        return self.rgn(radar_bev)  # (B, 1, H, W)
    

class BackwardProjection(nn.Module):
    """
    HyDRa backward projection: BEV → perspective → refine → BEV.
    
    Enforces consistency between BEV space and perspective view space.
    Radar attention weights (from RGN) guide which BEV regions to trust.
    """
    def __init__(self, bev_ch, feat_ch,
                 bev_h=128, bev_w=128,
                 x_range=(-50.0, 50.0),
                 z_range=(0.0, 100.0)):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_range = x_range
        self.z_range = z_range

        # Refine features in perspective space
        self.perspective_refine = nn.Sequential(
            nn.Conv2d(bev_ch, feat_ch, 3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU()
        )

        # Re-project back: perspective → BEV channels
        self.bev_restore = nn.Sequential(
            nn.Conv2d(feat_ch, bev_ch, 1),
            nn.BatchNorm2d(bev_ch),
            nn.ReLU()
        )

        # Final residual gate
        self.gate = nn.Sequential(
            nn.Conv2d(bev_ch * 2, bev_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, fused_bev, radar_attn, depth_prob, depth_values, cam_K):
        """
        fused_bev:    (B, C, bev_h, bev_w)   - SE-fused BEV
        radar_attn:   (B, 1, bev_h, bev_w)   - RGN attention weights
        depth_prob:   (B, D, H_img, W_img)   - depth distribution from DepthHead
        depth_values: (1, D, 1, 1)           - bin centers
        cam_K:        (B, 3, 3)              - camera intrinsics
        
        Returns:      (B, C, bev_h, bev_w)   - refined BEV
        """
        B, C, bev_H, bev_W = fused_bev.shape
        D = depth_prob.shape[1]
        device = fused_bev.device

        # ── 1. Weight BEV by radar confidence ──────────────────────────────
        # High-confidence radar regions (near detections) get stronger signal
        weighted_bev = fused_bev * radar_attn   # (B, C, bev_h, bev_w)

        # ── 2. Backward project BEV → perspective view ─────────────────────
        # For each image pixel, sample from BEV using geometry
        _, _, H_img, W_img = depth_prob.shape

        # Build pixel grid
        u = torch.linspace(0, W_img-1, W_img, device=device)
        v = torch.linspace(0, H_img-1, H_img, device=device)
        vv, uu = torch.meshgrid(v, u, indexing='ij')        # (H, W)
        uv1 = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)  # (H, W, 3)

        # Expected depth per pixel: sum(depth_prob * depth_values)
        depth_vals = depth_values.to(device)                # (1, D, 1, 1)
        expected_depth = (depth_prob * depth_vals).sum(dim=1)  # (B, H, W)

        # Back-project pixels to 3D using expected depth
        K_inv = torch.inverse(cam_K)                        # (B, 3, 3)
        uv1_flat = uv1.view(-1, 3).t()                      # (3, H*W)

        perspective_bev_feat = torch.zeros(B, C, H_img, W_img, device=device)

        for b in range(B):
            xyz = K_inv[b] @ uv1_flat                       # (3, H*W)
            z = expected_depth[b].view(-1)                  # (H*W,)
            xyz = xyz * z.unsqueeze(0)                      # (3, H*W) scale by depth

            x_3d = xyz[0]   # lateral
            z_3d = xyz[2]   # depth

            # Map to BEV grid coordinates
            x_norm = (x_3d - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
            z_norm = (z_3d - self.z_range[0]) / (self.z_range[1] - self.z_range[0])

            # Convert to grid_sample coordinates: [-1, 1]
            grid_x = x_norm * 2 - 1   # (H*W,)
            grid_z = z_norm * 2 - 1   # (H*W,)

            # Sample BEV features at the projected locations
            grid = torch.stack([grid_x, grid_z], dim=-1)   # (H*W, 2)
            grid = grid.view(1, H_img, W_img, 2)           # (1, H, W, 2)

            # grid_sample: sample weighted_bev at perspective-projected locations
            sampled = F.grid_sample(
                weighted_bev[b:b+1],    # (1, C, bev_h, bev_w)
                grid,                    # (1, H, W, 2)
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )   # (1, C, H, W)
            perspective_bev_feat[b] = sampled.squeeze(0)

        # ── 3. Refine in perspective space ──────────────────────────────────
        # Apply conv refinement — spatial context in image space
        refined_persp = self.perspective_refine(perspective_bev_feat)  # (B, feat_ch, H, W)

        # ── 4. Re-project refined features back to BEV ──────────────────────
        # Use same depth distribution to lift back up
        # (simplified: use depth-weighted scatter, same as ImageBEVEncoder)
        refined_persp_bev_ch = self.bev_restore(refined_persp)  # (B, C, H, W)

        # Lift back: outer product with depth_prob, scatter into BEV
        feat_expanded = refined_persp_bev_ch.unsqueeze(1).expand(B, D, C, H_img, W_img)
        depth_expanded = depth_prob.unsqueeze(2)                 # (B, D, 1, H, W)
        lifted = feat_expanded * depth_expanded                   # (B, D, C, H, W)

        # Rebuild BEV pixel grid indices (same as ImageBEVEncoder)
        u_lin = torch.linspace(0, W_img-1, W_img, device=device)
        v_lin = torch.linspace(0, H_img-1, H_img, device=device)
        vv2, uu2 = torch.meshgrid(v_lin, u_lin, indexing='ij')
        uv1_2 = torch.stack([uu2, vv2, torch.ones_like(uu2)], dim=-1)
        uv1_2 = uv1_2.view(1, 1, H_img, W_img, 3).expand(B, D, H_img, W_img, 3)

        K_inv_exp = torch.inverse(cam_K).view(B, 1, 1, 1, 3, 3)
        xyz_cam = torch.matmul(K_inv_exp, uv1_2.unsqueeze(-1)).squeeze(-1)
        xyz_cam = xyz_cam * depth_vals.view(1, D, 1, 1, 1)

        x_idx = xyz_cam[..., 0]
        z_idx = xyz_cam[..., 2]
        x_norm_bev = (x_idx - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        z_norm_bev = (z_idx - self.z_range[0]) / (self.z_range[1] - self.z_range[0])

        i = (z_norm_bev * bev_H).long().clamp(0, bev_H-1)
        j = (x_norm_bev * bev_W).long().clamp(0, bev_W-1)

        lifted_flat = lifted.reshape(B, D * H_img * W_img, C)
        flat_idx = i.reshape(B, D * H_img * W_img) * bev_W + \
                   j.reshape(B, D * H_img * W_img)

        reprojected_bev = torch.zeros(B, bev_H * bev_W, C, device=device)
        reprojected_bev = reprojected_bev.scatter_add_(
            1, flat_idx.unsqueeze(-1).expand(-1, -1, C), lifted_flat
        )
        counts = torch.zeros(B, bev_H * bev_W, device=device).scatter_add_(
            1, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32)
        )
        reprojected_bev = reprojected_bev / counts.clamp(min=1).unsqueeze(-1)
        reprojected_bev = reprojected_bev.view(B, bev_H, bev_W, C).permute(0, 3, 1, 2)

        # ── 5. Gated residual: blend original BEV with re-projected BEV ─────
        gate_weights = self.gate(
            torch.cat([fused_bev, reprojected_bev], dim=1)
        )   # (B, C, bev_h, bev_w) — learned blend weights
        return fused_bev + gate_weights * reprojected_bev
    
class RadarWeightedDepthConsistency(nn.Module):
    """
    Full HyDRa RDC module:
      1. SE-fuse image BEV + radar BEV
      2. RGN produces spatial radar attention weights
      3. Backward projection enforces BEV ↔ perspective consistency
    
    Drop-in replacement for your fusion_pX layers.
    """
    def __init__(self, img_bev_ch, radar_ch, out_ch,
                 bev_h=128, bev_w=128,
                 x_range=(-50.0, 50.0),
                 z_range=(0.0, 100.0)):
        super().__init__()

        self.se_fusion = SEFusionBlock(
            img_ch=img_bev_ch, radar_ch=radar_ch, out_ch=out_ch
        )
        self.rgn = RadarGuidanceNetwork(radar_ch=radar_ch)
        self.backward_proj = BackwardProjection(
            bev_ch=out_ch, feat_ch=out_ch,
            bev_h=bev_h, bev_w=bev_w,
            x_range=x_range, z_range=z_range
        )

    def forward(self, img_bev, radar_bev, depth_prob, depth_values, cam_K):
        """
        img_bev:      (B, img_bev_ch, H, W)
        radar_bev:    (B, radar_ch, H, W)
        depth_prob:   (B, D, H_img, W_img)
        depth_values: (1, D, 1, 1)
        cam_K:        (B, 3, 3)
        Returns:      (B, out_ch, H, W)
        """
        # Step 1: SE fusion
        fused = self.se_fusion(img_bev, radar_bev)     # (B, out_ch, H, W)

        # Step 2: Radar spatial attention
        radar_attn = self.rgn(radar_bev)               # (B, 1, H, W)

        # Step 3: Backward projection with radar guidance
        refined = self.backward_proj(
            fused, radar_attn, depth_prob, depth_values, cam_K
        )   # (B, out_ch, H, W)

        return refined
    





class BEVCrossAttention(nn.Module):
    """
    Cross-attention between image BEV (queries) and radar BEV (keys/values). 
    img_bev: (B, C_img, H, W)
    radar_bev: (B, C_rad, H, W)
    """
    def __init__(self, c_img, c_radar, n_heads=8, d_model=256):
        super().__init__()
        self.c_img = c_img
        self.c_radar = c_radar
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads ==0
        self.d_head = d_model // n_heads

        #Project image BEV to queries
        self.q_proj = nn.Linear(c_img, d_model)
        self.k_proj = nn.Linear(c_radar, d_model)
        self.v_proj = nn.Linear(c_radar, d_model)

        self.out_proj = nn.Linear(d_model, c_img)

    def forward(self, img_bev, radar_bev):
        B, C_img, H, W = img_bev.shape
        _, C_rad, Hr, Wr = radar_bev.shape
        assert H==Hr and W==Wr, "BEV grids must match spatially"

        #Flatten BEV to sequences
        q = img_bev.permute(0, 2, 3, 1).reshape(B, H*W, C_img)  # (B, HW, C_img)
        k = radar_bev.permute(0, 2, 3, 1).reshape(B, H*W, C_rad)    # (B, HW, C_rad)
        v = k

        #Linear projection
        q = self.q_proj(q)      # (B, HW, d_model)
        k = self.k_proj(k)      # (B, HW, d_model)
        v = self.v_proj(v)      # (B, HW, d_model)

        # Multi-head reshape
        B, N, _ = q.shape
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1,2)      # (B, h, N, d)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1,2)      # (B, h, N, d)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1,2)      # (B, h, N, d)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)   # (B, h, N, N)
        attn_weights = attn_scores.softmax(dim=-1)
        attn_out = torch.matmul(attn_weights, v)    # (B, h, N, d)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)   # (B, N, d_model)
        attn_out = self.out_proj(attn_out)      # (B, N, C_img)

        # Back to BEV grid
        attn_out = attn_out.view(B, H, W, C_img).permute(0, 3, 1, 2)    # (B, C_img, H, W)

        return attn_out 
    
class BEVWindowCrossAttention(nn.Module):
    """
    Local-window cross-attention between image BEV (queries) and radar BEV (keys/values).
    - img_bev:  (B, C_img, H, W)
    - radar_bev:(B, C_rad, H, W)
    Attention is computed within non-overlapping windows of size (Wh, Ww).
    """

    def __init__(self, c_img, c_radar, n_heads=4, d_model=256, window_size=(8, 8)):
        super().__init__()
        self.c_img = c_img
        self.c_radar = c_radar
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size  # (Wh, Ww)
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(c_img, d_model)
        self.k_proj = nn.Linear(c_radar, d_model)
        self.v_proj = nn.Linear(c_radar, d_model)
        self.out_proj = nn.Linear(d_model, c_img)

    def forward(self, img_bev, radar_bev):
        B, C_img, H, W = img_bev.shape
        _, C_rad, Hr, Wr = radar_bev.shape
        assert H == Hr and W == Wr, "BEV grids must match spatially"
        Wh, Ww = self.window_size
        assert H % Wh == 0 and W % Ww == 0, "H,W must be divisible by window size"

        # reshape into windows: (B, nWh, nWw, Wh, Ww, C)
        def to_windows(x, C):
            x = x.permute(0, 2, 3, 1)  # (B,H,W,C)
            x = x.view(B, H // Wh, Wh, W // Ww, Ww, C)  # (B, nWh, Wh, nWw, Ww, C)
            x = x.permute(0, 1, 3, 2, 4, 5)  # (B, nWh, nWw, Wh, Ww, C)
            return x

        q = to_windows(img_bev, C_img)
        k = to_windows(radar_bev, C_rad)
        v = k  # radar provides both keys and values

        # flatten window spatial dims: (B*nWh*nWw, Wh*Ww, C)
        B_, nWh, nWw, Wh, Ww, _ = q.shape
        Nw = Wh * Ww
        q = q.reshape(B_ * nWh * nWw, Nw, C_img)
        k = k.reshape(B_ * nWh * nWw, Nw, C_rad)
        v = v.reshape(B_ * nWh * nWw, Nw, C_rad)

        # linear projections
        q = self.q_proj(q)  # (Bwin, Nw, d_model)
        k = self.k_proj(k)  # (Bwin, Nw, d_model)
        v = self.v_proj(v)  # (Bwin, Nw, d_model)

        # multi-head split
        Bwin, N, _ = q.shape
        q = q.view(Bwin, N, self.n_heads, self.d_head).transpose(1, 2)  # (Bwin,h,N,d)
        k = k.view(Bwin, N, self.n_heads, self.d_head).transpose(1, 2)  # (Bwin,h,N,d)
        v = v.view(Bwin, N, self.n_heads, self.d_head).transpose(1, 2)  # (Bwin,h,N,d)

        # window attention: (Bwin,h,N,N)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_weights = attn_scores.softmax(dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (Bwin,h,N,d)

        # merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(Bwin, N, self.d_model)
        attn_out = self.out_proj(attn_out)  # (Bwin,N,C_img)

        # back to window grid: (B, nWh, nWw, Wh, Ww, C_img)
        attn_out = attn_out.view(B_, nWh, nWw, Wh, Ww, C_img)
        # merge back to (B, H, W, C_img)
        attn_out = attn_out.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B,nWh,Wh,nWw,Ww,C)
        attn_out = attn_out.view(B, H, W, C_img)
        attn_out = attn_out.permute(0, 3, 1, 2)  # (B,C_img,H,W)
        return attn_out


class SPPF(nn.Module):
    """Spatial Pyramid Pooling-Fast"""
    def __init__(self, in_c, out_c, k=5):
        super().__init__()
        c_ = in_c //2 # Hidden channels
        self.cv1 = ConvBNAct(in_c, c_, 1, 1)
        self.cv2 = ConvBNAct(c_*4, out_c, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1,padding = k//2)
    def forward(self, x):
        x=self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x,y1,y2,y3], dim=1))
    

class BEVScaleBackbone(nn.Module):
    """
    Applies spatial processing to a single BEV scale.
    Input:  (B, in_ch, H, W)
    Output: (B, out_ch, H, W)  - same spatial size
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch // 2
        
        # Downsample branch: compress spatial, force compact representation
        self.down1 = nn.Sequential(
            ConvBNAct(in_ch, mid_ch, k=3, s=2),   # H/2
            C2f(mid_ch, mid_ch, n=2)
        )
        self.down2 = nn.Sequential(
            ConvBNAct(mid_ch, out_ch, k=3, s=2),  # H/4
            C2f(out_ch, out_ch, n=2)
        )
        
        # Upsample branch: restore spatial detail
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNAct(out_ch, mid_ch, k=3, s=1)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNAct(mid_ch * 2, out_ch, k=3, s=1)  # *2 for skip concat
        )
        
        # Skip connection to match channels
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        
    def forward(self, x):
        skip = self.skip(x)                # (B, out_ch, H, W)
        
        d1 = self.down1(x)                 # (B, mid_ch, H/2, W/2)
        d2 = self.down2(d1)                # (B, out_ch, H/4, W/4)
        
        u2 = self.up2(d2)                  # (B, mid_ch, H/2, W/2)
        u2 = torch.cat([u2, d1], dim=1)   # (B, mid_ch*2, H/2, W/2)
        u1 = self.up1(u2)                  # (B, out_ch, H, W)
        
        return skip + u1                   # residual


class BEVMultiScaleNeck(nn.Module):
    """
    Fuses three BEV scales (128, 64, 32) into unified features for detection.
    Equivalent to SECOND neck but for your three-scale BEV.
    
    Inputs:
        p3: (B, 256, 128, 128)
        p4: (B, 256, 64, 64)
        p5: (B, 256, 32, 32)
    Outputs:
        out_p3: (B, 256, 128, 128)
        out_p4: (B, 256, 64, 64)
        out_p5: (B, 256, 32, 32)
    """
    def __init__(self, in_ch=256, out_ch=256):
        super().__init__()
        
        # Per-scale spatial backbone
        self.bev_backbone_p3 = BEVScaleBackbone(in_ch, out_ch)
        self.bev_backbone_p4 = BEVScaleBackbone(in_ch, out_ch)
        self.bev_backbone_p5 = BEVScaleBackbone(in_ch, out_ch)
        
        # Cross-scale fusion: coarser scales inform finer scales (top-down)
        # p5 → p4
        self.upsample_p5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNAct(out_ch, out_ch, k=1, s=1)
        )
        self.fuse_p4 = C2f(out_ch * 2, out_ch, n=1)
        
        # p4 → p3
        self.upsample_p4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNAct(out_ch, out_ch, k=1, s=1)
        )
        self.fuse_p3 = C2f(out_ch * 2, out_ch, n=1)

    def forward(self, p3, p4, p5):
        # Spatial backbone per scale
        p3 = self.bev_backbone_p3(p3)   # (B, 256, 128, 128)
        p4 = self.bev_backbone_p4(p4)   # (B, 256, 64, 64)
        p5 = self.bev_backbone_p5(p5)   # (B, 256, 32, 32)
        
        # Top-down: propagate global context from coarse to fine
        p4 = self.fuse_p4(torch.cat([p4, self.upsample_p5(p5)], dim=1))
        p3 = self.fuse_p3(torch.cat([p3, self.upsample_p4(p4)], dim=1))
        
        return p3, p4, p5
    
class ResNet101Backbone(nn.Module):
    def __init__(self, pretrained=True, out_channels=(256, 512, 1024)):
        super().__init__()

        weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet101(weights=weights)

        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool
        )

        self.layer1 = net.layer1   # stride 4
        self.layer2 = net.layer2   # stride 8
        self.layer3 = net.layer3   # stride 16
        self.layer4 = net.layer4   # stride 32

        self.proj_p3 = nn.Conv2d(512,  out_channels[0], kernel_size=1)
        self.proj_p4 = nn.Conv2d(1024, out_channels[1], kernel_size=1)
        self.proj_p5 = nn.Conv2d(2048, out_channels[2], kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        p3 = self.layer2(x)   # B, 512, H/8,  W/8
        p4 = self.layer3(p3)  # B, 1024, H/16, W/16
        p5 = self.layer4(p4)  # B, 2048, H/32, W/32

        p3 = self.proj_p3(p3) # B, 256, H/8,  W/8
        p4 = self.proj_p4(p4) # B, 512, H/16, W/16
        p5 = self.proj_p5(p5) # B, 1024, H/32, W/32

        return p3, p4, p5
    
class YOLOv8Backbone(nn.Module):
    def __init__(self, in_c=3, base_c=64, repeats=[1, 2, 3, 1], sppf_c=1024):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(in_c, base_c, k=3, s=2),  # /2
            ConvBNAct(base_c, base_c*2, k=3, s=2) # /4
        )
        # Stage 1 (P3, output stride 8)
        self.c2f_1 = C2f(base_c*2, base_c*2, n=repeats[0])
        self.conv1 = ConvBNAct(base_c*2, base_c*4, k=3, s=2, p=1)
        
        # Stage 2 (P4, output stride 16)
        self.c2f_2 = C2f(base_c*4, base_c*4, n=repeats[1])
        self.conv2 = ConvBNAct(base_c*4, base_c*8,k=3, s=2, p=1)
        
        # Stage 3 (P5, output stride 32)
        self.c2f_3 = C2f(base_c*8, base_c*8, n=repeats[2])
        self.conv3 = ConvBNAct(base_c*8, base_c*16, k=3, s=2, p=1)
        self.c2f_4 = C2f(base_c*16, base_c*16, n=repeats[3])
        
        self.sppf = SPPF(base_c*16, sppf_c)
    def forward(self, x):
        x = self.stem(x)
        p1 = self.c2f_1(x)
        p2 = self.conv1(p1)
        p3 = self.c2f_2(p2)
        p4 = self.conv2(p3)
        p4 = self.c2f_3(p4)
        p5 = self.conv3(p4)
        p5 = self.c2f_4(p5)
        p5 = self.sppf(p5)
        #print(f"P3 shape: {p3.shape}")
        #print(f"P4 shape: {p4.shape}")
        #print(f"P5 shape: {p5.shape}")
        return [p3, p4, p5]  # outputs for P3 (stride8), P4 (stride16), P5 (stride32)
    
class YOLOv8Neck(nn.Module):
    """
    YOLOv8-style neck with FPN (top-down) + PAN (bottom-up) pathway.
    
    Input:  p3 (B, 256, 80, 80), p4 (B, 512, 40, 40), p5 (B, 1024, 20, 20)
    Output: Three detection-ready features at different scales
            - P3: (B, 256, 80, 80) for small objects
            - P4: (B, 256, 40, 40) for medium objects  
            - P5: (B, 256, 20, 20) for large objects
    """
    def __init__(self, ch_in=[256, 512, 1024], pan_out=256):
        super().__init__()
        
        # ============ Channel reduction layers ============
        # Reduce all feature maps to uniform channel count
        self.reduce_p5 = nn.Conv2d(ch_in[2], pan_out, 1)  # 1024 -> 256
        self.reduce_p4 = nn.Conv2d(ch_in[1], pan_out, 1)  # 512 -> 256
        self.reduce_p3 = nn.Conv2d(ch_in[0], pan_out, 1)  # 256 -> 256
        
        # ============ Top-Down FPN pathway ============
        # Upsample and fuse features from coarse to fine
        self.c2f_p4_td = C2f(pan_out * 2, pan_out, n=2)  # After concat: 512 -> 256
        self.c2f_p3_td = C2f(pan_out * 2, pan_out, n=2)  # After concat: 512 -> 256
        
        # ============ Bottom-Up PAN pathway ============
        # Downsample and fuse features from fine to coarse
        self.down_p3 = nn.Conv2d(pan_out, pan_out, 3, stride=2, padding=1)  # 128->64
        self.c2f_p4_bu = C2f(pan_out * 2, pan_out, n=2)  # 512 -> 256
        
        self.down_p4 = nn.Conv2d(pan_out, pan_out, 3, stride=2, padding=1)  # 64->32
        self.c2f_p5_bu = C2f(pan_out * 2, pan_out, n=2)  # 512 -> 256
        
    def forward(self, features):
        """
        Args:
            features: tuple of (p3, p4, p5)
                p3: (B, 256, 128, 128)  - fine-grained features
                p4: (B, 512, 64, 64)    - medium-grained features  
                p5: (B, 1024, 32, 32)   - coarse-grained features
        
        Returns:
            List of [out_p3, out_p4, out_p5] for multi-scale detection
                out_p3: (B, 256, 128, 128) - detect small objects
                out_p4: (B, 256, 64, 64)   - detect medium objects
                out_p5: (B, 256, 32, 32)   - detect large objects
        """
        p3, p4, p5 = features
        
        # ============ Channel Reduction ============
        # Bring all feature maps to same channel dimension (256)
        # Spatial sizes remain: P5=20×20, P4=40×40, P3=80×80
        fp5 = self.reduce_p5(p5)  # (B, 256, 20, 20)  ✅ FIXED
        fp4 = self.reduce_p4(p4)  # (B, 256, 40, 40)  ✅ FIXED
        fp3 = self.reduce_p3(p3)  # (B, 256, 80, 80) ✅ CORRECT

        # print(f"After channel reduction:")
        # print(f"  fp5: {fp5.shape}")  # (B, 256, 20, 20)
        # print(f"  fp4: {fp4.shape}")  # (B, 256, 40, 40)
        # print(f"  fp3: {fp3.shape}")  # (B, 256, 80, 80)
        
        # ============ Top-Down FPN: Coarse -> Fine ============
        # Enrich fine-scale features with coarse-scale semantic info
        
        # P5 -> P4: Upsample P5 (20×20 -> 40x40) and fuse with P4
        up_p5 = F.interpolate(fp5, size=fp4.shape[2:], mode='bilinear', align_corners=True)  # ✅ 20x20 -> 40x40
        p4_concat = torch.cat([up_p5, fp4], dim=1)  # (B, 512, 40, 40) ✅
        fp4_td = self.c2f_p4_td(p4_concat)  # (B, 256, 40, 40) ✅
        
        # P4 -> P3: Upsample P4 (40x40 -> 80x80) and fuse with P3  
        up_p4 = F.interpolate(fp4_td, size=fp3.shape[2:], mode='bilinear', align_corners=True)  # ✅ 40x40 -> 80x80
        p3_concat = torch.cat([up_p4, fp3], dim=1)  # (B, 512, 80, 80) ✅
        fp3_td = self.c2f_p3_td(p3_concat)  # (B, 256, 80, 80) ✅
        
        # ============ Bottom-Up PAN: Fine -> Coarse ============
        # Enrich coarse-scale features with fine-scale localization info
        
        # P3 -> P4: Downsample P3 (80x80 -> 40x40) and fuse with P4
        down_p3 = self.down_p3(fp3_td)  # (B, 256, 40, 40) ✅
        p4_concat_bu = torch.cat([down_p3, fp4_td], dim=1)  # (B, 512, 40, 40) ✅
        fp4_bu = self.c2f_p4_bu(p4_concat_bu)  # (B, 256, 40, 40) ✅
        
        # P4 -> P5: Downsample P4 (40x40 -> 20x20) and fuse with P5
        down_p4 = self.down_p4(fp4_bu)  # (B, 256, 20, 20) ✅
        p5_concat_bu = torch.cat([down_p4, fp5], dim=1)  # (B, 512, 20, 20) ✅
        fp5_bu = self.c2f_p5_bu(p5_concat_bu)  # (B, 256, 20, 20) ✅
        
        # print(f"Final outputs:")
        # print(f"  fp3_td: {fp3_td.shape}")  # (B, 256, 80, 80)
        # print(f"  fp4_bu: {fp4_bu.shape}")  # (B, 256, 40, 40)
        # print(f"  fp5_bu: {fp5_bu.shape}")  # (B, 256, 20, 20)
        
        
        # Return multi-scale outputs for detection heads
        # P3: 80×80 grid (1.25 m/cell) - small/close objects
        # P4: 40×40 grid   (2.50 m/cell) - medium objects
        # P5: 20x20 grid   (5.0 m/cell) - large/distant objects
        return [fp3_td, fp4_bu, fp5_bu]





class YOLOv8Head(nn.Module):
    """
    YOLOv8-style decoupled detection head for 3D object detection.
    
    Outputs per scale:
        - Objectness: (B, 1, H, W) - probability that cell contains object center
        - Box regression: (B, 7, H, W) - [x_off, y, z_off, w_log, l_log, h_log, yaw]
        - Classification: (B, num_classes, H, W) - class logits
        - IOU Prediction: (B, 1, H, W) -  predicted IOU quality
    
    Three heads operate at:
        - Head 0: 128×128 grid (small objects)
        - Head 1: 64×64 grid (medium objects)
        - Head 2: 32×32 grid (large objects)
    """
    def __init__(self, in_ch=[256, 256, 256], num_classes=12):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = len(in_ch)
        
        # Separate branches for classification and regression (decoupled head)
        self.obj_branches = nn.ModuleList()
        self.cls_branches = nn.ModuleList()
        self.reg_branches = nn.ModuleList()
        self.iou_branches = nn.ModuleList()
        
        for c in in_ch:

            # self.obj_branches.append(
            #     nn.Sequential(
            #         ConvBNAct(c, c, 3, 1),
            #         nn.Conv2d(c, 1, 1)
            #     )
            # )
            self.obj_branches.append(
                nn.Sequential(
                    nn.Conv2d(c, c, 3, 1, padding=1, groups=c),  # Depthwise
                    nn.BatchNorm2d(c),
                    nn.SiLU(),
                    nn.Conv2d(c, 1, 1)
                )
            )
            # Classification branch: predicts objectness + class scores
            self.cls_branches.append(
                nn.Sequential(
                    ConvBNAct(c, c//2, 3, 1),
                    ConvBNAct(c//2, c//4, 3, 1),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(c//4, num_classes, 1)  # [cls1, cls2, ...]
                )
            )
            
            # Regression branch: predicts 3D box parameters
            self.reg_branches.append(
                nn.Sequential(
                    ConvBNAct(c, c//2, 3, 1),
                    ConvBNAct(c//2, c//4, 3, 1),
                    nn.Conv2d(c//4, 7, 1)  # [x_off, y, z_off, w_log, l_log, h_log, yaw]
                )
            )

            # IOU prediction branch (shares features with regression)
            self.iou_branches.append(
                nn.Sequential(
                    ConvBNAct(c, c//2, 3, 1),
                    nn.Conv2d(c//2, 1, 1)   # Predicts IOU quality
                )
            )

        # ⭐ ADD BIAS INIT HERE - after objbranches created
        with torch.no_grad():
            for i, branch in enumerate(self.obj_branches):
                final_conv = branch[-1]  # Last layer (1x1 Conv2d)
                if hasattr(final_conv, 'bias') and final_conv.bias is not None:
                    # P3(small objs): -2.0 | P4(medium): -2.3 | P5(large): -2.6
                    bias_val = [-2.0, -2.3, -2.6][i]  
                    final_conv.bias.fill_(bias_val)
                    print(f"P{3+i} obj bias init: {bias_val:.2f}")
    
    def forward(self, features):
        """
        Args:
            features: List of [p3, p4, p5]
                p3: (B, 256, 128, 128)
                p4: (B, 256, 64, 64)
                p5: (B, 256, 32, 32)
        
        Returns:
            outputs: List of predictions for each scale
                Each prediction: (B, 1+num_classes+7+1, H, W)   +1 for IOU quality
                Channel layout: [obj, x_off, y, z_off, w_log, l_log, h_log, yaw, cls...]
        """
        outputs = []
        
        for i, feat in enumerate(features):
            # Decoupled prediction
            obj_out = self.obj_branches[i](feat)  # (B, 1, H, W)
            cls_out = self.cls_branches[i](feat)  # (B, num_classes, H, W)
            reg_out = self.reg_branches[i](feat)  # (B, 7, H, W)
            iou_out = self.iou_branches[i](feat)  # (B, 1, H, W)
            
            B, _, H, W = feat.shape
            
            # Split classification outputs
            obj_logits = obj_out      # (B, 1, H, W) - objectness logits
            cls_logits = cls_out        # (B, num_classes, H, W) - class logits
            
            # Split regression outputs (keep as raw values for loss computation)
            x_off = reg_out[:, 0:1, :, :]             # (B, 1, H, W) - x offset (will be sigmoid in loss)
            y_raw = reg_out[:, 1:2, :, :]             # (B, 1, H, W) - y metric (will be normalized in loss)
            z_off = reg_out[:, 2:3, :, :]             # (B, 1, H, W) - z offset (will be sigmoid in loss)
            w_log = reg_out[:, 3:4, :, :]             # (B, 1, H, W) - log(width)
            l_log = reg_out[:, 4:5, :, :]             # (B, 1, H, W) - log(length)
            h_log = reg_out[:, 5:6, :, :]             # (B, 1, H, W) - log(height)
            yaw_raw = reg_out[:, 6:7, :, :]           # (B, 1, H, W) - yaw in radians
            
            # Concatenate: [obj, x_off, y, z_off, w_log, l_log, h_log, yaw, cls...]
            # Note: Keep as raw logits/values for training stability
            # Activations applied in loss function during decoding
            raw_output = torch.cat([
                obj_logits,   # Channel 0: objectness (logit)
                x_off,        # Channel 1: x offset (raw, sigmoid in loss)
                y_raw,        # Channel 2: y coordinate (raw, normalized in loss)
                z_off,        # Channel 3: z offset (raw, sigmoid in loss)
                w_log,        # Channel 4: log(w) (raw, exp in loss)
                l_log,        # Channel 5: log(l) (raw, exp in loss)
                h_log,        # Channel 6: log(h) (raw, exp in loss)
                yaw_raw,      # Channel 7: yaw (raw, used directly)
                cls_logits,
                iou_out    # Channels 8+: class logits
            ], dim=1)  # (B, 1+7+num_classes, H, W)
            
            outputs.append(raw_output)
        
        return outputs  # List of [out_p3, out_p4, out_p5]



class FusedYOLO(nn.Module):
    def __init__(self, radar_dim = 7, num_classes=12):
        super().__init__()
        self.num_seg_classes = num_classes
        #print("before backbone")


        # self.backbone = YOLOv8Backbone()
        self.backbone = ResNet101Backbone(pretrained=True, out_channels=(256, 512, 1024))
        self.neck = YOLOv8Neck(ch_in=[256, 512, 1024], pan_out=256)

        

        self.sparse_proposals = SparseSemanticProposals(
            num_classes=num_classes,
            foreground_classes=[6,7,8,9,10,11],
            confidence_threshold=0.3,
            max_peaks_per_class=20,
            peak_kernel_size=7
        )

        self.proposal_bev_encoder_p3 = SparseProposalToBEV(
            bev_h=128, bev_w=128,
            proposal_dim=64, 
            bev_channels=64,
            x_range=(-50.0, 50.0),
            z_range=(0.0, 100.0),
            gaussian_sigma=2.0
        )

        self.proposal_bev_encoder_p4 = SparseProposalToBEV(
            bev_h=64, bev_w=64,
            proposal_dim=64, 
            bev_channels=64,
            x_range=(-50.0, 50.0),
            z_range=(0.0, 100.0),
            gaussian_sigma=2.0
        )

        self.proposal_bev_encoder_p5 = SparseProposalToBEV(
            bev_h=32, bev_w=32,
            proposal_dim=64, 
            bev_channels=64,
            x_range=(-50.0, 50.0),
            z_range=(0.0, 100.0),
            gaussian_sigma=2.0
        )

        self.radar_painter = RadarSemanticPainter()

        self.radar_bev_encoder = RadarFeatureEncoderMultiScale(
            grid_sizes=[(128, 128),(64, 64), (32,32)],
            input_dim=7+num_classes, 
            out_channels=(256, 256, 256), 
            x_range=(X_MIN, X_MAX), 
            z_range=(Z_MIN, Z_MAX)
            )
        
        # self.radar_bev_encoder = RadarFeatureEncoder(grid_size=(BEV_H, BEV_W), input_dim=7+num_classes, out_channels=(256, ), x_range=(X_MIN, X_MAX), z_range=(Z_MIN, Z_MAX))
        
        self.img_depth_head_p3 = ImageDepthHead(in_ch=256, num_depth_bins=DEPTH_BINS, feat_ch=256)
        self.img_bev_encoder_p3 = ImageBEVEncoder(feat_ch=256, num_depth_bins=DEPTH_BINS, bev_h=128, bev_w=128, x_range=(X_MIN, X_MAX), z_range=(Z_MIN, Z_MAX),num_seg_classes=num_classes)
        

        # self.radar_img_attn_p3 = BEVWindowCrossAttention(
        #     c_img=256 + num_classes,   # image BEV channels
        #     c_radar=256,               # radar BEV channels
        #     n_heads=4,
        #     d_model=256,
        #     window_size=(8, 8)         # 8×8 windows on 128×128 grid
        # )
        # self.fusion_p3 = nn.Sequential(
        #     nn.Conv2d((256 + num_classes) + 64, 256, 1),  # collapse channels after attn+proposal
        #     nn.BatchNorm2d(256), nn.ReLU()
        # )
        # Fuse: Image BEV (256+12) + Radar BEV(256) + Proposal BEV (64)
        fusion_input_ch_p3 = (256 + num_classes) + 256 +64
        self.fusion_p3 = nn.Sequential(
            nn.Conv2d(fusion_input_ch_p3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # self.bev_fusion_p3 = BEVFusionBlock(img_ch=256+num_classes, radar_ch=256, out_ch=256)
        
        self.img_depth_head_p4 = ImageDepthHead(in_ch=256, num_depth_bins=DEPTH_BINS, feat_ch=256)
        self.img_bev_encoder_p4 = ImageBEVEncoder(feat_ch=256, num_depth_bins=DEPTH_BINS, bev_h=64, bev_w=64, x_range=(X_MIN, X_MAX), z_range=(Z_MIN, Z_MAX), num_seg_classes=num_classes)
         
        # self.radar_img_attn_p4 = BEVWindowCrossAttention(
        #     c_img=256 + num_classes,   # image BEV channels
        #     c_radar=256,               # radar BEV channels
        #     n_heads=4,
        #     d_model=256,
        #     window_size=(4, 4)         # 8×8 windows on 128×128 grid
        # )
        # self.fusion_p4 = nn.Sequential(
        #     nn.Conv2d((256 + num_classes) + 64, 256, 1),  # collapse channels after attn+proposal
        #     nn.BatchNorm2d(256), nn.ReLU() 
        # )
         # Fuse: Image BEV (512+12) + Radar BEV(512) + Proposal BEV (64)
        fusion_input_ch_p4 = (256 + num_classes) + 256 +64
        self.fusion_p4 = nn.Sequential(
            nn.Conv2d(fusion_input_ch_p4, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.img_depth_head_p5 = ImageDepthHead(in_ch=256, num_depth_bins=DEPTH_BINS, feat_ch=256)
        self.img_bev_encoder_p5 = ImageBEVEncoder(feat_ch=256, num_depth_bins=DEPTH_BINS, bev_h=32, bev_w=32, x_range=(X_MIN, X_MAX), z_range=(Z_MIN, Z_MAX), num_seg_classes=num_classes)
        
        # self.radar_img_attn_p5 = BEVWindowCrossAttention(
        #     c_img=256 + num_classes,   # image BEV channels
        #     c_radar=256,               # radar BEV channels
        #     n_heads=4,
        #     d_model=256,
        #     window_size=(4, 4)         # 8×8 windows on 128×128 grid
        # )
        # self.fusion_p5 = nn.Sequential(
        #     nn.Conv2d((256 + num_classes) + 64, 256, 1),  # collapse channels after attn+proposal
        #     nn.BatchNorm2d(256), nn.ReLU() 
        # )

         # Fuse: Image BEV (1024+12) + Radar BEV(1025) + Proposal BEV (64)
        fusion_input_ch_p5 = (256 + num_classes) + 256 +64
        self.fusion_p5 = nn.Sequential(
            nn.Conv2d(fusion_input_ch_p5, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # self.rdc_p3 = RadarWeightedDepthConsistency(
        #     img_bev_ch=256 + num_classes,  # 268
        #     radar_ch=256,
        #     out_ch=256,
        #     bev_h=128, bev_w=128
        # )
        # self.rdc_p4 = RadarWeightedDepthConsistency(
        #     img_bev_ch=256 + num_classes,
        #     radar_ch=256,
        #     out_ch=256,
        #     bev_h=64, bev_w=64
        # )
        # self.rdc_p5 = RadarWeightedDepthConsistency(
        #     img_bev_ch=256 + num_classes,
        #     radar_ch=256,
        #     out_ch=256,
        #     bev_h=32, bev_w=32
        
        self.bev_neck=BEVMultiScaleNeck(in_ch=256, out_ch=256)
        
        self.head = YOLOv8Head(in_ch=[256, 256, 256], num_classes=num_classes)
        # self.neck = SimpleBEVNeck(in_ch=256, out_ch=256)
        # self.head = YOLOv8Head(in_ch=[256], num_classes=num_classes)
        #print("Done with head init")
    
    def forward(self, img, radar, cam_K, seg_mask=None):
        # img: (B, 3, H, W)
        # radar: (B, N, radar_dim)
        # cam_K: (B, 3, 3)
        # cam_T: (B, 4, 4)  I want centers to be in camera and rotation in cam_ego frame
        # Extract image features
        B = img.shape[0]
        device = img.device
        
        if seg_mask is not None:
            radar_painted = self.radar_painter(radar, seg_mask, cam_K)  # (B, N, 7+C)
        else:
            radar_painted = radar
        radar_feats = self.radar_bev_encoder(radar_painted)     # (B, 256, 128, 128), (B, 256, 64, 64), (B, 256, 32, 32)

        img_feats= self.backbone(img)                   # (B, 256, H3, W3), (B, 512, H4, W4), (B, 1024, H5, W5)

        if self.neck is not None:
            img_feats = self.neck(img_feats)        # (B, 256, H3, W3), (B, 256, H4, W4), (B, 256, H5, W5)

        img_pan3, img_pan4, img_pan5 = img_feats   
        
        radar_bev_p3, radar_bev_p4, radar_bev_p5 = radar_feats
        # radar_bev_p3 = self.radar_bev_encoder(radar_painted)[0]     # (B, 256, BEV_H, BEV_W)
        # print(f"Radar BEV sizes:")
        # print(f"  radar_bev_p3: {radar_bev_p3.shape}")  # Should be (B, 256, 128, 128)
        # print(f"  radar_bev_p4: {radar_bev_p4.shape}")  # Should be (B, 512, 64, 64)
        # print(f"  radar_bev_p5: {radar_bev_p5.shape}")  # Should be (B, 1024, 32, 32)

        # Lift the segmentation masks
        seg_p3 = F.interpolate(seg_mask, size=img_pan3.shape[2:], mode='bilinear')
        img_feat_p3, depth_prob_p3, depth_values_p3 = self.img_depth_head_p3(img_pan3, cam_K)
        img_bev_p3 = self.img_bev_encoder_p3(img_feat_p3, depth_prob_p3, depth_values_p3, cam_K, seg_p3)      # (B, 256, 128, 128)

        proposals_p3, xyz_p3 = self.sparse_proposals(seg_p3, depth_prob_p3, depth_values_p3, cam_K, stride=8)
        proposal_bev_p3 = self.proposal_bev_encoder_p3(proposals_p3, xyz_p3, B, device)

        # img_bev_p3_refined = self.radar_img_attn_p3(img_bev_p3, radar_bev_p3)
        # fused_bev_p3 = self.fusion_p3(torch.cat([img_bev_p3_refined, proposal_bev_p3], dim=1))
        fused_bev_p3 = self.fusion_p3(torch.cat([img_bev_p3, radar_bev_p3, proposal_bev_p3], dim=1))
    

        seg_p4 = F.interpolate(seg_mask, size=img_pan4.shape[2:], mode='bilinear')
        img_feat_p4, depth_prob_p4, depth_values_p4 = self.img_depth_head_p4(img_pan4, cam_K)
        img_bev_p4 = self.img_bev_encoder_p4(img_feat_p4, depth_prob_p4, depth_values_p4, cam_K, seg_p4)      # (B, 512, 64, 64)

        proposals_p4, xyz_p4 = self.sparse_proposals(seg_p4, depth_prob_p4, depth_values_p4, cam_K, stride=16)
        proposal_bev_p4 = self.proposal_bev_encoder_p4(proposals_p4, xyz_p4, B, device)

        # img_bev_p4_refined = self.radar_img_attn_p4(img_bev_p4, radar_bev_p4)
        # fused_bev_p4 = self.fusion_p4(torch.cat([img_bev_p4_refined, proposal_bev_p4], dim=1))

        fused_bev_p4 = self.fusion_p4(torch.cat([img_bev_p4, radar_bev_p4, proposal_bev_p4], dim=1))


        seg_p5 = F.interpolate(seg_mask, size=img_pan5.shape[2:], mode='bilinear')
        img_feat_p5, depth_prob_p5, depth_values_p5 = self.img_depth_head_p5(img_pan5, cam_K)
        img_bev_p5 = self.img_bev_encoder_p5(img_feat_p5, depth_prob_p5, depth_values_p5, cam_K, seg_p5)       # (B, 1024, 32, 32)

        proposals_p5, xyz_p5 = self.sparse_proposals(seg_p5, depth_prob_p5, depth_values_p5, cam_K, stride=32)
        proposal_bev_p5 = self.proposal_bev_encoder_p5(proposals_p5, xyz_p5, B, device)

        # img_bev_p5_refined = self.radar_img_attn_p5(img_bev_p5, radar_bev_p5)
        # fused_bev_p5 = self.fusion_p5(torch.cat([img_bev_p5_refined, proposal_bev_p5], dim=1))

        fused_bev_p5 = self.fusion_p5(torch.cat([img_bev_p5, radar_bev_p5, proposal_bev_p5], dim=1))

        #print("p5 shape: ", len(p5))
        # features = [fused_bev_p3, fused_bev_p4, fused_bev_p5]       # (B, 256, 128, 128), (B, 512, 64, 64), (B, 1024, 32, 32)
        # print("Backbone output:")
        # print("P3 size: ", fused_bev_p3.shape)
        # print("P4 size: ", fused_bev_p4.shape)
        # print("P5 size: ", fused_bev_p5.shape)
        # features = [p3_bev]
        fused_bev_p3, fused_bev_p4, fused_bev_p5 = self.bev_neck(
            fused_bev_p3, fused_bev_p4, fused_bev_p5
        )
        out = self.head([fused_bev_p3, fused_bev_p4, fused_bev_p5])
        
        return out
    

#model = FusedYOLO(num_classes=5)
#print(model)
#for name, param in model.named_parameters():
#    print(name, param.requires_grad)