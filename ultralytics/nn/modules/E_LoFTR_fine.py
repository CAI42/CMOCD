import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

# 假设的CoarseMatching类，参考现有代码
class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        self.temperature = config['dsmax_temperature']
        self.skip_softmax = config['skip_softmax']
        self.fp16matmul = config['fp16matmul']
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5, [feat_c0, feat_c1])

        if self.fp16matmul:
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
            if mask_c0 is not None:
                sim_matrix = sim_matrix.masked_fill(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(), -1e4)
        else:
            with torch.autocast(enabled=False, device_type='cuda'):
                sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
                if mask_c0 is not None:
                    sim_matrix = sim_matrix.float().masked_fill(
                        ~(mask_c0[..., None] * mask_c1[:, None]).bool(), -1e9)

        if self.skip_softmax:
            sim_matrix = sim_matrix
        else:
            sim_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        data.update({'conf_matrix': sim_matrix})
        data.update(**self.get_coarse_match(sim_matrix, data))
        return data

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix.device
        mask = conf_matrix > self.thr
        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c', **axes_lengths)
        if 'mask0' not in data:
            self.mask_border(mask, self.border_rm, False)
        else:
            self.mask_border_with_padding(mask, self.border_rm, False, data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)', **axes_lengths)

        mask = mask * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        if self.training:
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
            else:
                num_candidates_max = self.compute_max_candidates(data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(num_matches_pred, (num_matches_train - self.train_pad_num_gt_min,), device=_device)

            gt_pad_indices = torch.randint(len(data['spv_b_ids']), (max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min),), device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']], [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack([i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]], dim=1) * scale0
        mkpts1_c = torch.stack([j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]], dim=1) * scale1
        m_bids = b_ids[mconf != 0]
        coarse_matches.update({
            'm_bids': m_bids,
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })
        return coarse_matches

    def mask_border(self, m, b, v):
        if b <= 0:
            return
        m[:, :b] = v
        m[:, :, :b] = v
        m[:, :, :, :b] = v
        m[:, :, :, :, :b] = v
        m[:, -b:] = v
        m[:, :, -b:] = v
        m[:, :, :, -b:] = v
        m[:, :, :, :, -b:] = v

    def mask_border_with_padding(self, m, bd, v, p_m0, p_m1):
        if bd <= 0:
            return
        m[:, :bd] = v
        m[:, :, :bd] = v
        m[:, :, :, :bd] = v
        m[:, :, :, :, :bd] = v
        h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
        h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
        for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
            m[b_idx, h0 - bd:] = v
            m[b_idx, :, w0 - bd:] = v
            m[b_idx, :, :, h1 - bd:] = v
            m[b_idx, :, :, :, w1 - bd:] = v

    def compute_max_candidates(self, p_m0, p_m1):
        h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
        h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
        max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
        return max_cand


# 假设的FineMatching类，参考现有代码
class FineMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_regress_temperature = config['match_fine']['local_regress_temperature']
        self.local_regress_slicedim = config['match_fine']['local_regress_slicedim']
        self.fp16 = config['half']
        self.validate = False

    def forward(self, feat_0, feat_1, data):
        M, WW, C = feat_0.shape
        W = int(WW ** 0.5)
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        if M == 0:
            assert self.training == False, "M is always > 0 while training, see coarse_matching.py"
            data.update({
                'conf_matrix_f': torch.empty(0, WW, WW, device=feat_0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        with torch.autocast(enabled=True if not (self.training or self.validate) else False, device_type='cuda'):
            feat_f0, feat_f1 = feat_0[..., :-self.local_regress_slicedim], feat_1[..., :-self.local_regress_slicedim]
            feat_ff0, feat_ff1 = feat_0[..., -self.local_regress_slicedim:], feat_1[..., -self.local_regress_slicedim:]
            feat_f0, feat_f1 = feat_f0 / C ** 0.5, feat_f1 / C ** 0.5
            conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1)
            conf_matrix_ff = torch.einsum('mlc,mrc->mlr', feat_ff0, feat_ff1 / (self.local_regress_slicedim) ** 0.5)

        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
        softmax_matrix_f = softmax_matrix_f.reshape(M, self.WW, self.W + 2, self.W + 2)
        softmax_matrix_f = softmax_matrix_f[..., 1:-1, 1:-1].reshape(M, self.WW, self.WW)

        if self.training or self.validate:
            data.update({'sim_matrix_ff': conf_matrix_ff})
            data.update({'conf_matrix_f': softmax_matrix_f})

        self.get_fine_ds_match(softmax_matrix_f, data)

        idx_l, idx_r = data['idx_l'], data['idx_r']
        m_ids = torch.arange(M, device=idx_l.device, dtype=torch.long).unsqueeze(-1)
        m_ids = m_ids[:len(data['mconf'])]
        idx_r_iids, idx_r_jids = idx_r // W, idx_r % W

        m_ids, idx_l, idx_r_iids, idx_r_jids = m_ids.reshape(-1), idx_l.reshape(-1), idx_r_iids.reshape(-1), idx_r_jids.reshape(-1)
        delta = torch.stack(torch.meshgrid(torch.arange(3), torch.arange(3), indexing='ij'), dim=-1).to(conf_matrix_ff.device).to(torch.long)
        m_ids = m_ids[..., None, None].expand(-1, 3, 3)
        idx_l = idx_l[..., None, None].expand(-1, 3, 3)

        idx_r_iids = idx_r_iids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_r_jids = idx_r_jids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 0]

        if idx_l.numel() == 0:
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        conf_matrix_ff = conf_matrix_ff.reshape(M, self.WW, self.W + 2, self.W + 2)
        conf_matrix_ff = conf_matrix_ff[m_ids, idx_l, idx_r_iids, idx_r_jids]
        conf_matrix_ff = conf_matrix_ff.reshape(-1, 9)
        conf_matrix_ff = F.softmax(conf_matrix_ff / self.local_regress_temperature, -1)
        heatmap = conf_matrix_ff.reshape(-1, 3, 3)

        coords_normalized = F.conv2d(heatmap.unsqueeze(1), torch.ones(1, 1, 3, 3, device=heatmap.device), padding=1).squeeze(1)
        coords_normalized = coords_normalized / coords_normalized.sum(dim=(1, 2), keepdim=True)
        coords_normalized = torch.stack([
            (coords_normalized * torch.arange(3, device=coords_normalized.device).unsqueeze(0).unsqueeze(1)).sum(dim=(1, 2)),
            (coords_normalized * torch.arange(3, device=coords_normalized.device).unsqueeze(0).unsqueeze(2)).sum(dim=(1, 2))
        ], dim=1)
        coords_normalized = (coords_normalized - 1) / 1

        if data['bs'] == 1:
            scale1 = scale * data['scale1'] if 'scale0' in data else scale
        else:
            scale1 = scale * data['scale1'][data['b_ids']][:len(data['mconf']), ...][:, None, :].expand(-1, -1, 2).reshape(-1, 2) if 'scale0' in data else scale

        self.get_fine_match_local(coords_normalized, data, scale1)
        return data

    def get_fine_match_local(self, coords_normed, data, scale1):
        mkpts0_c, mkpts1_c = data['mkpts0_c'], data['mkpts1_c']
        mkpts0_f = mkpts0_c
        mkpts1_f = mkpts1_c + (coords_normed * (3 // 2) * scale1)
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })

    @torch.no_grad()
    def get_fine_ds_match(self, conf_matrix, data):
        m, _, _ = conf_matrix.shape
        conf_matrix = conf_matrix.reshape(m, -1)[:len(data['mconf']), ...]
        val, idx = torch.max(conf_matrix, dim=-1)
        idx = idx[:, None]
        idx_l, idx_r = idx // self.WW, idx % self.WW
        data.update({'idx_l': idx_l, 'idx_r': idx_r})

        grid = torch.stack(torch.meshgrid(torch.arange(self.W), torch.arange(self.W), indexing='ij'), dim=-1).to(conf_matrix.device) - self.W // 2 + 0.5
        grid = grid.reshape(1, -1, 2).expand(m, -1, -1)
        delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
        delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))

        scale0 = self.scale * data['scale0'][data['b_ids']] if 'scale0' in data else self.scale
        scale1 = self.scale * data['scale1'][data['b_ids']] if 'scale0' in data else self.scale

        if torch.is_tensor(scale0) and scale0.numel() > 1:
            mkpts0_f = (data['mkpts0_c'][:, None, :] + (delta_l * scale0[:len(data['mconf']), ...][:, None, :])).reshape(-1, 2)
            mkpts1_f = (data['mkpts1_c'][:, None, :] + (delta_r * scale1[:len(data['mconf']), ...][:, None, :])).reshape(-1, 2)
        else:
            mkpts0_f = (data['mkpts0_c'][:, None, :] + (delta_l * scale0)).reshape(-1, 2)
            mkpts1_f = (data['mkpts1_c'][:, None, :] + (delta_r * scale1)).reshape(-1, 2)

        data.update({
            "mkpts0_c": mkpts0_f,
            "mkpts1_c": mkpts1_f
        })


# 新的Fuse1类，结合粗匹配和精细匹配
class E_LoFTR_Fuse(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_matching = FineMatching(config)

    def forward(self,x):
        rgb = x[0]
        t = x[1]
    
    def forward(self, feat_c0, feat_c1, feat_f0_unfold, feat_f1_unfold, data, mask_c0=None, mask_c1=None):
        # 粗匹配
        data = self.coarse_matching(feat_c0, feat_c1, data, mask_c0, mask_c1)

        # 精细匹配
        data = self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        return data
