import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops import knn_points, knn_gather

from .encode import EncodeNet
from .base import Conv1D, PileConv


class PointEncoder(nn.Module):
    def __init__(self, knn, use_w=True):
        super(PointEncoder, self).__init__()
        self.use_w = use_w

        self.encodeNet = EncodeNet(num_convs=2,
                                    in_channels=3,
                                    conv_channels=24,
                                    num_fc_layers=3,
                                    knn=knn,
                                )
        d_code = self.encodeNet.out_channels   # 60

        dim_1 = 128
        self.conv_1 = Conv1D(d_code, dim_1)

        self.pconv1 = PileConv(dim_1, dim_1*2)
        self.pconv2 = PileConv(dim_1*2, dim_1*2)
        self.pconv3 = PileConv(dim_1*2, dim_1*2)
        self.pconv4 = PileConv(dim_1*2, dim_1*2)
        self.pconv5 = PileConv(dim_1*2, dim_1*2)

        self.conv_2 = Conv1D(dim_1*2, 256)
        self.conv_3 = Conv1D(256, 128)

        if self.use_w:
            self.alpha = nn.Parameter(torch.Tensor(1,), requires_grad=True)
            self.beta = nn.Parameter(torch.Tensor(1,), requires_grad=True)
            nn.init.ones_(self.alpha.data)
            nn.init.ones_(self.beta.data)

    def forward(self, pos, num_pcl, knn_idx):
        """
            pos:  (B, 3, N)
            knn_idx: (B, N, K)
        """
        ### Encoding
        data_tuple = None
        y = self.encodeNet(x=pos, pos=pos, knn_idx=knn_idx)                # (B, C, N)
        y = self.conv_1(y)

        if self.use_w:
            ### compute distance weights from points to its center point (ref: FKAConv, POCO)
            dist = torch.sqrt((pos.detach() ** 2).sum(dim=1))              # (B, N)
            dist_w = torch.sigmoid(-self.alpha * dist + self.beta)
            dist_w_s = dist_w.sum(dim=1, keepdim=True)                     # (B, 1)
            dist_w_s = dist_w_s + (dist_w_s == 0) + 1e-6
            dist_w = (dist_w / dist_w_s * dist.shape[1]).unsqueeze(1)      # (B, 1, N)

            data_tuple = (dist_w[:, :, :num_pcl//2], y[:, :, :num_pcl//2])
        else:
            dist_w = None
            data_tuple = (dist_w, y[:, :, :num_pcl//2])

        ### decrease the number of points and pile the features
        y1 = self.pconv1(y, num_pcl*2, dist_w=dist_w)                      # (B, C, n*2)
        y2 = self.pconv2(y1, num_pcl, dist_w=dist_w)                       # (B, C, n)
        y2 = y2 + y1[:, :, :num_pcl]

        y3 = self.pconv3(y2, num_pcl, dist_w=dist_w)
        y4 = self.pconv4(y3, num_pcl//2, dist_w=dist_w)
        y4 = y4 + y3[:, :, :num_pcl//2]

        y5 = self.pconv5(y4, num_pcl//2, dist_w=dist_w)

        y = self.conv_3(self.conv_2(y5))                                  # (B, C, n)

        return y, data_tuple


class GVO(nn.Module):
    def __init__(self, num_pat=1, encode_knn=16):
        super(GVO, self).__init__()
        self.num_pcl = num_pat // 4
        self.encode_knn = encode_knn

        self.pointEncoder = PointEncoder(knn=self.encode_knn, use_w=True)

        self.norm_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.conv_p = Conv1D(128, 128)
        self.pconv_1 = PileConv(128, 128)
        self.pconv_2 = PileConv(128, 128)
        self.pconv_3 = PileConv(128, 128)

        self.conv_1 = Conv1D(128, 128)
        self.conv_n = Conv1D(128, 128)
        self.conv_w = nn.Conv1d(128, 1, 1)

        self.loss_function = nn.L1Loss()

    def forward(self, pcl_pat, query_vectors=None, mode_test=False):
        """
            pcl_pat: (B, N, 3)
            pcl_sample: (B, N', 3), N' < N
            query_vectors: (B, M, 3)
        """
        _, knn_idx, _ = knn_points(pcl_pat, pcl_pat, K=self.encode_knn+1, return_nn=False)  # (B, N, K+1)

        ### Encoder
        pcl_pat = pcl_pat.transpose(2, 1)                                 # (B, 3, N)
        y, data_tuple = self.pointEncoder(pcl_pat,
                                        num_pcl=self.num_pcl,
                                        knn_idx=knn_idx[:,:,1:self.encode_knn+1],
                                    )                                     # (B, C, n)
        dw, _ = data_tuple                                                # (B, C, n)

        y0 = self.conv_p(y)
        y1 = self.pconv_1(y0, self.num_pcl//2, dist_w=dw)
        y2 = self.pconv_2(y1, self.num_pcl//4, dist_w=dw)
        y2 = y2 + y1[:, :, :self.num_pcl//4] + y0[:, :, :self.num_pcl//4]
        feat = self.pconv_3(y2, self.num_pcl//4, dist_w=dw)
        feat = self.conv_1(feat)                                          # (B, C, n=N//16)

        feat_vec = self.norm_encoder(query_vectors)                       # (B, M, C)

        ### Output
        weights = 0.01 + torch.sigmoid(self.conv_w(feat))                 # (B, 1, n)
        feat_w = self.conv_n(feat * weights)
        feat_w = feat_w.max(dim=2, keepdim=False)[0]                      # (B, C)
        feat_w = feat_w.unsqueeze(1).repeat(1, query_vectors.shape[1], 1) # (B, M, C)

        feat_cat = torch.cat([feat_w, feat_vec], dim=2)                   # (B, M, 128+128)
        angle_pred = self.mlp_out(feat_cat).squeeze(-1)                   # (B, M)

        if mode_test:
            return angle_pred

        return angle_pred, weights

    def get_loss(self, q_target=None, pred_weights=None, angle_pred=None, angle_gt=None, pcl_in=None):
        """
            q_target: query point normal, (B, 3)
            pred_weights: patch point weight, (B, 1, N)
            pcl_in: input (noisy) point clouds, (B, N, 3)
        """
        angle_loss = 0.5 * self.loss_function(angle_pred, angle_gt)

        ### compute true_weight by fitting distance
        if pred_weights is not None:
            pred_weights = pred_weights.squeeze()
            num_out = pred_weights.shape[1]
            pcl_local = pcl_in[:, :num_out, :] - pcl_in[:, 0:1, :]                                      # (B, N, 3)
            scale = (pcl_local ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0]      # (B, 1, 1)
            pcl_local = pcl_local / scale

            ### the distances between the neighbor points and ground truth tangent plane
            gamma = 0.3
            thres_d = 0.05 ** 2
            normal_dis = torch.bmm(q_target.unsqueeze(1), pcl_local.transpose(2, 1)).pow(2).squeeze()   # (B, N), dis^2
            sigma = torch.mean(normal_dis, dim=1) * gamma + 1e-5                                        # (B,)
            threshold_matrix = torch.ones_like(sigma) * thres_d                                         # (B,)
            sigma = torch.where(sigma < thres_d, threshold_matrix, sigma)                               # (B,), all sigma >= thres_d
            true_weights_plane = torch.exp(-1 * torch.div(normal_dis, sigma.unsqueeze(-1)))             # (B, N), -dis/mean(dis') -> (-∞, 0) -> (0, 1)

            true_weights = true_weights_plane
            weight_loss = (true_weights - pred_weights).pow(2).mean()

        loss = angle_loss + weight_loss
        return loss, (angle_loss, weight_loss)



class MLPNet_linear(nn.Module):
    def __init__(self,
                 d_aug=3,
                 d_mid=256,
                 d_code=0,
                 d_out=1,
                 n_mid=8,
                 bias=0.5,
                 geometric_init=True,
                 inside_grad=True,
            ):
        super(MLPNet_linear, self).__init__()
        assert n_mid > 3
        dims = [d_aug] + [d_mid for _ in range(n_mid)] + [d_out]
        self.num_layers = len(dims)
        self.skip_in = [n_mid // 2]
        self.d_code = d_code

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - d_aug - d_code
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    if inside_grad:
                        nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        nn.init.constant_(lin.bias, bias)
                    else:
                        nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        nn.init.constant_(lin.bias, -bias)
                else:
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    nn.init.constant_(lin.bias, 0.0)

            setattr(self, "lin" + str(l), lin)

    def forward(self, pos, code=None):
        """
            pos: (*, N, C)
        """
        x = pos
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                if self.d_code > 0:
                    x = torch.cat([x, code, pos], dim=-1)
                else:
                    x = torch.cat([x, pos], dim=-1)
                x = x / np.sqrt(2)

            lin = getattr(self, "lin" + str(l))
            x = lin(x)

            if l < self.num_layers - 2:
                x = F.relu(x)
        return x

    def gradient(self, x, code=None):
        """
            x: (*, N, C)
        """
        x.requires_grad = True
        y = self.forward(pos=x, code=code)         # (*, N, 1), signed distance

        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        grads = torch.autograd.grad(outputs=y,
                                    inputs=x,
                                    grad_outputs=d_output,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
        grad_norm = F.normalize(grads, dim=-1)     # (*, N, 3)
        return y, grad_norm


class NGL(nn.Module):
    def __init__(self):
        super(NGL, self).__init__()
        self.net = MLPNet_linear(d_aug=3, d_mid=256, d_code=0, d_out=1, n_mid=8)

    def forward(self, pcl_source, code=None):
        """
            pcl_source: (*, N, 3)
        """
        with torch.set_grad_enabled(True):
            self.sd, self.grad_norm = self.net.gradient(x=pcl_source, code=code)   # (*, N, 1), (*, N, 3)

        return self.grad_norm

    def get_loss(self, pcl_raw=None, pcl_source=None, knn_idx=None):
        """
            pcl_raw: (*, M, 3), M >= N
            pcl_source: (*, N, 3)
            knn_idx: (*, L, K)
        """
        frames = knn_gather(pcl_raw, knn_idx)    # (B, L, K, 3)
        v = pcl_source.unsqueeze(2) - frames
        v = v.mean(2)                            # (B, N, 3)

        loss = torch.linalg.norm((v - self.sd * self.grad_norm), ord=2, dim=-1).mean()

        return loss

