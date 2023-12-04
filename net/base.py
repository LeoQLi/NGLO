import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points


def cos_angle(v1, v2):
    """
        V1, V2: (N, 3)
        return: (N,)
    """
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)


def get_sign(pred, min_val=-1.0):
    p = pred >= 0.0         # logits to bool
    sign = torch.full_like(p, min_val, dtype=torch.float32)
    sign[p] = 1.0           # bool to sign factor
    return sign


def knn_group_v2(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, F, N)
    :param  idx:    (B, M, k)
    :return (B, F, M, k)
    """
    B, F, N = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(2).expand(B, F, M, N)
    idx = idx.unsqueeze(1).expand(B, F, M, k)

    return torch.gather(x, dim=3, index=idx)


def knn_group(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)

    return torch.gather(x, dim=2, index=idx)


def gather(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M)
    :return (B, M, F)
    """
    B, N, F = tuple(x.size())
    _, M    = tuple(idx.size())
    idx = idx.unsqueeze(2).expand(B, M, F)
    return torch.gather(x, dim=1, index=idx)


def get_knn_idx_dist(pos:torch.FloatTensor, query:torch.FloatTensor, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
            knn_dist: (B, M, k)
    """
    B, N, F = tuple(pos.size())
    M = query.size(1)

    pos = pos.unsqueeze(1).expand(B, M, N, F)
    query  = query.unsqueeze(2).expand(B, M, N, F)                # B * M * N * F
    dist = torch.sum((pos - query) ** 2, dim=3, keepdim=False)    # B * M * N
    knn_idx = torch.argsort(dist, dim=2)[:, :, offset:k+offset]   # B * M * k
    knn_dist = torch.gather(dist, dim=2, index=knn_idx)           # B * M * k

    return knn_idx, knn_dist


def get_knn_idx(pos, query, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    _, knn_idx, _ = knn_points(pos, query, K=k+offset, return_nn=False)
    return knn_idx[:, :, offset:]


def square_distance(src, dst):
    """ Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm;
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    # # [B, N, M, C] = [B, N, 1, C] - [B, 1, M, C]
    # diffs = src[:, :, None, :] - dst[:, None, :, :]
    # diffs = torch.sum(diffs ** 2, dim=-1)

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1).contiguous())
    dist += torch.sum(src ** 2, dim=-1)[:, :, None]
    dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    return dist


class Conv1D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=True, with_relu=True):
        super(Conv1D, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv1d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
            x: (B, C, N)
        """
        if self.with_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.with_relu:
            x = F.relu(x)
        return x


class PileConv(nn.Module):
    def __init__(self, input_dim, output_dim, with_mid=True):
        super(PileConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.with_mid = with_mid

        self.conv_in = Conv1D(input_dim, input_dim*2, with_bn=True, with_relu=with_mid)

        if with_mid:
            self.conv_mid = Conv1D(input_dim*2, input_dim//2, with_bn=True)
            self.conv_out = Conv1D(input_dim + input_dim//2, output_dim, with_bn=True)
        else:
            self.conv_out = Conv1D(input_dim + input_dim*2, output_dim, with_bn=True)

    def forward(self, x, num_out, dist_w=None):
        """
            x: (B, C, N)
        """
        BS, _, N = x.shape

        if dist_w is None:
            y = self.conv_in(x)
        else:
            y = self.conv_in(x * dist_w[:,:,:N])        # (B, C*2, N)

        feat_g = torch.max(y, dim=2, keepdim=True)[0]   # (B, C*2, 1)
        if self.with_mid:
            feat_g = self.conv_mid(feat_g)              # (B, C/2, 1)

        x = torch.cat([x[:, :, :num_out],
                    feat_g.view(BS, -1, 1).repeat(1, 1, num_out),
                    ], dim=1)

        x = self.conv_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim):
        super(Attention, self).__init__()
        self.conv_q = nn.Conv1d(in_dim1, out_dim, 1)
        self.conv_k = nn.Conv1d(in_dim2, out_dim, 1)
        self.conv_v = nn.Conv1d(in_dim2, out_dim, 1)
        self.conv_attn = nn.Conv1d(out_dim, out_dim, 1)
        self.conv_a = nn.Conv1d(out_dim+in_dim1, 128, 1)

    def forward(self, x, y, w=None):
        """
        x: (B, C, N)
        y: (B, C, N) or (B, C, 1)
        """
        query = self.conv_q(x)
        key   = self.conv_k(y)
        value = self.conv_v(y)
        attn = self.conv_attn(query + key)
        attn = torch.softmax(attn, dim=2)
        agg = attn * value
        if w is not None:
            agg = agg * w
        agg = torch.cat([agg, x], dim=1)
        agg = self.conv_a(agg)
        return agg


def cangle(vec1, vec2):
    n = vec1.norm(p=2, dim=-1)*vec2.norm(p=2, dim=-1)
    mask = (n < 1e-8).float()
    cang = (1-mask)*(vec1*vec2).sum(-1)/(n+mask)
    return cang


def compute_prf(pos, normals, edge_idx, scale=10.0):
    row, col = edge_idx
    d = pos[col] - pos[row]
    normals1 = normals[row]
    normals2 = normals[col]
    ppf = torch.stack([cangle(normals1, d), cangle(normals2, d),
                       cangle(normals1, normals2), torch.sqrt((d**2).sum(-1))*scale], dim=-1)
    return ppf


if torch.cuda.is_available():
    import quat_to_mat
class QuatToMat(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        assert torch.cuda.is_available()
        assert x.size(-1) == 4
        self.save_for_backward(x)
        return quat_to_mat.quat_to_mat_fw(x)

    @staticmethod
    def backward(self, g_y):
        x, = self.saved_variables
        g_x = None

        if self.needs_input_grad[0]:
            g_x = quat_to_mat.quat_to_mat_bw(x, g_y)
        return g_x