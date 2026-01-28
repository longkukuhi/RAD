import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import os

def ssim_loss_func(img1, img2, window_size=11):
    """
    calculate SSIM Loss (1 - SSIM).
    input shape: [N, 3, H, W]
    """
    channel = img1.size(1)

    def gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32)
        coords = coords - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g  # [size]

    # 1D kernel: [1, window_size]
    _1d = gaussian_window(window_size, 1.5).unsqueeze(0)  # [1, W]

    # 2D kernel via outer product: [W, W]
    _2d = _1d.t() @ _1d  # [W, W]

    window = _2d.unsqueeze(0).unsqueeze(0)  # [1, 1, W, W]
    window = window.to(img1.device).expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1 - ssim_map.mean()

class ViTill(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
    ) -> None:
        super(ViTill, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):

        en_list = self.encoder.get_intermediate_layers(x, n=self.target_layers, norm=False)

        side = int(math.sqrt(en_list[0].shape[1]))

        x = self.fuse_feature(en_list)

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        return en, de


    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + 1 + self.encoder.num_register_tokens,
                              h * w + 1 + self.encoder.num_register_tokens, device=device)
        mask_all[1 + self.encoder.num_register_tokens:, 1 + self.encoder.num_register_tokens:] = mask
        return mask_all


class ViTMae(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        embed_dim,
        patch_size=16,
        in_chans=3,
        norm_pix_loss=False,
    ) -> None:
        super(ViTMae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.norm_pix_loss = norm_pix_loss
        
        # Normalization layer before the prediction head
        self.decoder_norm = nn.LayerNorm(embed_dim)
        
        # Decoder to patch predictor
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True) 

        # Handle register tokens if not present in encoder
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.encoder.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.encoder.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # Calculate mean loss on removed patches only
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_loss_full_image(self, imgs, pred):
        """
        Calculates loss over the entire image (masked and unmasked patches).
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L]
        loss = loss.mean()        # Mean loss over all patches
        return loss

    def forward_loss_hybrid(self, imgs, pred, mask, alpha=0.5, beta=0.5):
        """
        Weighted combination of masked and unmasked loss.
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3] 
        mask: [N, L]
        alpha: masked loss weight (Though implementation seems to hardcode 0.5 weight on unmasked)
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) 

        # masked loss
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        # unmasked loss
        unmasked_loss = (loss * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6)

        pred_img = self.unpatchify(pred)

        #ssim loss
        loss_ssim = ssim_loss_func(pred_img, imgs)

        return masked_loss + alpha * unmasked_loss + beta * loss_ssim

    def visualize_mae(self, imgs, recon, masks, save_path=None):
        """
        Visualization utility.
        imgs: [B, 3, H, W]
        recon: [B, 3, H, W]
        masks: [B, L]
        """
        imgs = imgs.detach().cpu()
        recon = recon.detach().cpu()
        masks = masks.detach().cpu()

        # ImageNet statistics for denormalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        imgs_vis = imgs * std + mean
        recon_vis = recon * std + mean
        
        imgs_vis = imgs_vis.clamp(0, 1)
        recon_vis = recon_vis.clamp(0, 1)

        B, _, H, W = imgs.shape
        p = self.encoder.patch_embed.patch_size[0]
        h = w = H // p

        # Reshape mask to image dimensions for visualization
        masks = masks.view(B, h, w).unsqueeze(1).float()
        masks = F.interpolate(masks, size=(H, W), mode="nearest")  # [B,1,H,W]
        
        masked_imgs = imgs_vis * (1 - masks)

        B_show = min(4, B)
        fig, ax = plt.subplots(B_show, 3, figsize=(12, 4 * B_show))

        for i in range(B_show):
            # Original
            ax[i, 0].imshow(TF.to_pil_image(imgs_vis[i]))
            ax[i, 0].set_title("Original (Transformed Input)")
            ax[i, 0].axis("off")
            
            # Masked
            ax[i, 1].imshow(TF.to_pil_image(masked_imgs[i]))
            ax[i, 1].set_title("Masked")
            ax[i, 1].axis("off")
            
            # Reconstruction
            ax[i, 2].imshow(TF.to_pil_image(recon_vis[i]))
            ax[i, 2].set_title("Reconstruction")
            ax[i, 2].axis("off")

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    def reconstruct_full_coverage(self, x, target_mask_ratio: float = 0.5):
        """
        Testing only: Full Coverage Parallel Reconstruction
        - Each patch is masked exactly once across the splits.
        - Uses only model predictions; does not mix in original pixel values.
        Args:
            x: [B, C, H, W]
            target_mask_ratio: Target mask ratio per pass, e.g., 0.0625
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # ---- Derive patch count using encoder's patch_embed ----
        with torch.no_grad():
            patches = self.encoder.patch_embed(x)   # [B, H_p, W_p, C_e]
            _, H_p, W_p, _ = patches.shape
            
        L = H_p * W_p

        # Patch size (used only for upsampling pixel-level masks)
        p = self.encoder.patch_embed.patch_size
        if isinstance(p, (tuple, list)):
            p = p[0]

        # ---- 1. Calculate num_splits ----
        target_splits = int(1 / target_mask_ratio)  # Ideal number of splits, e.g., 16

        factors = [i for i in range(1, L + 1) if L % i == 0]
        num_splits = min(factors, key=lambda v: abs(v - target_splits))

        actual_ratio = 1.0 / num_splits
        patch_per_split = L // num_splits
        # You can print the actual ratio here for debugging
        # print(f"L={L}, num_splits={num_splits}, actual_ratio={actual_ratio:.2%}")

        # ---- 2. Generate mutually exclusive random groups for each sample ----
        # rand_perm: [B, L], a random permutation for each sample
        rand_perm = torch.rand(B, L, device=device).argsort(dim=1)
        # Split into [B, num_splits, patch_per_split]
        perm_groups = rand_perm.view(B, num_splits, patch_per_split)

        # ---- 3. Construct super batch ----
        x_rep = x.repeat_interleave(num_splits, dim=0)  # [B*num_splits, C, H, W]

        flat_indices = perm_groups.view(B * num_splits, patch_per_split)  # [B*num_splits, patch_per_split]

        mask_bool = torch.zeros(B * num_splits, L, device=device, dtype=torch.bool)
        mask_bool.scatter_(1, flat_indices, True)  # True indicates masked

        # ---- 4. Parallel reconstruction using encoder + decoder ----
        feat_dict = self.encoder.forward_features(x_rep, masks=mask_bool)
        latent = feat_dict["x_norm_patchtokens"]   # [B*num_splits, L, C_e]

        de = latent
        for blk in self.decoder:
            de = blk(de)
        de = self.decoder_norm(de)
        de = self.decoder_pred(de)                 # [B*num_splits, L, p^2*3]

        recon_imgs = self.unpatchify(de)           # [B*num_splits, 3, H, W]

        # ---- 5. Reassemble into a single complete reconstruction image ----
        # Upsample mask from patch level to pixel level
        mask_pixel = mask_bool.view(B * num_splits, 1, H_p, W_p).float()
        mask_pixel = F.interpolate(mask_pixel, size=(H, W), mode="nearest")  # [B*num_splits,1,H,W]

        recon_parts = recon_imgs * mask_pixel      # Keep only the masked regions
        recon_parts = recon_parts.view(B, num_splits, 3, H, W)

        # Since splits are mutually exclusive in patch dimension, simply sum them up
        recon_final = recon_parts.sum(dim=1)       # [B,3,H,W]

        return recon_final

    def forward(self, x, is_training=True, full_cov_eval=False, target_mask_ratio=0.5):
        if is_training:
            en_list = self.encoder.forward_mae_center_mask(
                x,
                mask_ratio=target_mask_ratio,
                is_training=True,
            )
            en = en_list["x_norm_patchtokens"]

            de = en
            for blk in self.decoder:
                de = blk(de)
            de = self.decoder_norm(de)
            de = self.decoder_pred(de)

            loss = self.forward_loss(x, de, en_list["mae_mask"])
            return loss
        else:
            if full_cov_eval:
                return self.reconstruct_full_coverage(
                    x,
                    target_mask_ratio=target_mask_ratio,
                )
            else:
                en_list = self.encoder.forward_mae_center_mask(x, is_training=False)
                en = en_list["x_norm_patchtokens"]
                de = en
                for blk in self.decoder:
                    de = blk(de)
                de = self.decoder_norm(de)
                de = self.decoder_pred(de)
                recon = self.unpatchify(de)
                return recon


class ViTillCat(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[1, 3, 5, 7],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
    ) -> None:
        super(ViTillCat, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        for i, blk in enumerate(self.decoder):
            x = blk(x)

        en = [torch.cat([en_list[idx] for idx in self.fuse_layer_encoder], dim=2)]
        de = [x]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

class ViTAD(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 5, 8, 11],
            fuse_layer_encoder=[0, 1, 2],
            fuse_layer_decoder=[2, 5, 8],
            mask_neighbor_size=0,
            remove_class_token=False,
    ) -> None:
        super(ViTAD, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
            x = x[:, 1 + self.encoder.num_register_tokens:, :]

        # x = torch.cat(en_list, dim=2)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [en_list[idx] for idx in self.fuse_layer_encoder]
        de = [de_list[idx] for idx in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de


class ViTillv2(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7]
    ) -> None:
        super(ViTillv2, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en.append(x)

        x = self.fuse_feature(en)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de.append(x)

        side = int(math.sqrt(x.shape[1]))

        en = [e[:, self.encoder.num_register_tokens + 1:, :] for e in en]
        de = [d[:, self.encoder.num_register_tokens + 1:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        return en[::-1], de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class ViTillv3(nn.Module):
    def __init__(
            self,
            teacher,
            student,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_dropout=0.,
    ) -> None:
        super(ViTillv3, self).__init__()
        self.teacher = teacher
        self.student = student
        if fuse_dropout > 0:
            self.fuse_dropout = nn.Dropout(fuse_dropout)
        else:
            self.fuse_dropout = nn.Identity()
        self.target_layers = target_layers
        if not hasattr(self.teacher, 'num_register_tokens'):
            self.teacher.num_register_tokens = 0

    def forward(self, x):
        with torch.no_grad():
            patch = self.teacher.prepare_tokens(x)
            x = patch
            en = []
            for i, blk in enumerate(self.teacher.blocks):
                if i <= self.target_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.target_layers:
                    en.append(x)
            en = self.fuse_feature(en, fuse_dropout=False)

        x = patch
        de = []
        for i, blk in enumerate(self.student):
            x = blk(x)
            if i in self.target_layers:
                de.append(x)
        de = self.fuse_feature(de, fuse_dropout=False)

        en = en[:, 1 + self.teacher.num_register_tokens:, :]
        de = de[:, 1 + self.teacher.num_register_tokens:, :]
        side = int(math.sqrt(en.shape[1]))

        en = en.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        de = de.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        return [en.contiguous()], [de.contiguous()]

    def fuse_feature(self, feat_list, fuse_dropout=False):
        if fuse_dropout:
            feat = torch.stack(feat_list, dim=1)
            feat = self.fuse_dropout(feat).mean(dim=1)
            return feat
        else:
            return torch.stack(feat_list, dim=1).mean(dim=1)


class ReContrast(nn.Module):
    def __init__(
            self,
            encoder,
            encoder_freeze,
            bottleneck,
            decoder,
    ) -> None:
        super(ReContrast, self).__init__()
        self.encoder = encoder
        self.encoder.layer4 = None
        self.encoder.fc = None

        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.layer4 = None
        self.encoder_freeze.fc = None

        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x):
        en = self.encoder(x)
        with torch.no_grad():
            en_freeze = self.encoder_freeze(x)
        en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]
        de = self.decoder(self.bottleneck(en_2))
        de = [a.chunk(dim=0, chunks=2) for a in de]
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]
        return en_freeze + en, de

    def train(self, mode=True, encoder_bn_train=True):
        self.training = mode
        if mode is True:
            if encoder_bn_train:
                self.encoder.train(True)
            else:
                self.encoder.train(False)
            self.encoder_freeze.train(False)  # the frozen encoder is eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
        else:
            self.encoder.train(False)
            self.encoder_freeze.train(False)
            self.bottleneck.train(False)
            self.decoder.train(False)
        return self


def update_moving_average(ma_model, current_model, momentum=0.99):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = update_average(old_weight, up_weight)

    for current_buffers, ma_buffers in zip(current_model.buffers(), ma_model.buffers()):
        old_buffer, up_buffer = ma_buffers.data, current_buffers.data
        ma_buffers.data = update_average(old_buffer, up_buffer, momentum)


def update_average(old, new, momentum=0.99):
    if old is None:
        return new
    return old * momentum + (1 - momentum) * new


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
