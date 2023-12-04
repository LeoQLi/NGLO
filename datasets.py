import os
import sys
import copy
import time
import random
import numpy as np
from tqdm.auto import tqdm
import scipy.spatial as spatial
import torch
from torch.utils.data import Dataset


# All PCPNet shapes
all_train_sets = ['fandisk100k', 'bunny100k', 'armadillo100k', 'dragon_xyzrgb100k', 'boxunion_uniform100k',
                  'tortuga100k', 'flower100k', 'Cup33100k']
all_test_sets = ['galera100k', 'icosahedron100k', 'netsuke100k', 'Cup34100k', 'sphere100k',
                 'cylinder100k', 'star_smooth100k', 'star_halfsmooth100k', 'star_sharp100k', 'Liberty100k',
                 'boxunion2100k', 'pipe100k', 'pipe_curve100k', 'column100k', 'column_head100k',
                 'Boxy_smooth100k', 'sphere_analytic100k', 'cylinder_analytic100k', 'sheet_analytic100k']

all_FamousShape = ['3DBenchy', 'angel', 'dragon', 'dragon2', 'hand', 'happy', 'horse', 'LibertyBase', 'lucy', 'serapis', 'statuette', 'teapot']



def load_data(filedir, filename, dtype=np.float32, wo=False):
    d = None
    filepath = os.path.join(filedir, 'npy', filename + '.npy')
    os.makedirs(os.path.join(filedir, 'npy'), exist_ok=True)
    if os.path.exists(filepath):
        if wo:
            return True
        d = np.load(filepath)
    else:
        d = np.loadtxt(os.path.join(filedir, filename), dtype=dtype)
        np.save(filepath, d)
    return d


def normalization(pcl):
    """
        pcl: (N, 3)
    """
    shape_scale = np.max([np.max(pcl[:,0])-np.min(pcl[:,0]), np.max(pcl[:,1])-np.min(pcl[:,1]), np.max(pcl[:,2])-np.min(pcl[:,2])])
    shape_center = [(np.max(pcl[:,0])+np.min(pcl[:,0]))/2, (np.max(pcl[:,1])+np.min(pcl[:,1]))/2, (np.max(pcl[:,2])+np.min(pcl[:,2]))/2]
    pcl = pcl - shape_center
    pcl = pcl / shape_scale
    return pcl


def normalization_1(pcl):
    """
        pcl: (N, 3)
    """
    pcl = pcl - pcl.mean(axis=0, keepdims=True)
    scale = (1 / np.abs(pcl).max()) * 0.999999
    pcl = pcl * scale
    return pcl


def spherical_sample(num):
    x = np.random.randn(num, 3)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class PCATrans(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        # compute PCA of points in the patch, center the patch around the mean
        pts = data['pcl_pat']
        pts_mean = pts.mean(0)
        pts = pts - pts_mean

        trans, _, _ = torch.svd(torch.t(pts))  # (3, 3)
        pts = torch.mm(pts, trans)

        cp_new = -pts_mean # since the patch was originally centered, the original cp was at (0,0,0)
        cp_new = torch.matmul(cp_new, trans)

        # re-center on original center point
        data['pcl_pat'] = pts - cp_new
        data['pca_trans'] = trans

        if 'normal_center' in data:
            data['normal_center'] = torch.matmul(data['normal_center'], trans)
        if 'normal_pat' in data:
            data['normal_pat'] = torch.matmul(data['normal_pat'], trans)
        if 'query_vectors' in data:
            data['query_vectors'] = torch.matmul(data['query_vectors'], trans)
        if 'pcl_sample' in data:
            data['pcl_sample'] = torch.matmul(data['pcl_sample'], trans)
        if 'sample_near' in data:
            data['sample_near'] = torch.matmul(data['sample_near'], trans)
        if 'normal_sample' in data:
            data['normal_sample'] = torch.matmul(data['normal_sample'], trans)
        return data


class AddNoise(object):
    def __init__(self):
        super().__init__()
        self.scale_list = [0.0012, 0.006, 0.012]   # as in PCPNet

    def __call__(self, data):
        scale = random.choice(self.scale_list)
        boundingbox_diagonal = np.linalg.norm(data['pcl'].max(0) - data['pcl'].min(0))
        scale = scale * boundingbox_diagonal
        noise = np.random.normal(0.0, scale=scale, size=data['pcl'].shape).astype(np.float32)
        # clip = scale
        # noise = np.clip(noise, a_min=-clip, a_max=clip)
        data['pcl'] = data['pcl'] + noise

        # noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        # data['pcl_noisy'] = data['pcl'] + torch.randn_like(data['pcl']) * noise_std
        # data['noise_std'] = noise_std
        return data


class AddUniformBallNoise_Normal_1(object):
    def __init__(self, scale=0.0, num=0):
        super().__init__()
        self.num = num

    def apply(self, v):
        ### quaternion (Shoemake, K., 1992)
        q = np.random.normal(size=(self.num, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        rot = spatial.transform.Rotation.from_quat(q) # (x, y, z, w)
        v = rot.apply(v)
        return v

    def __call__(self, v):
        return self.apply(v)


class AddUniformBallNoise_Normal(object):
    def __init__(self, scale=0.0, num=0):
        super().__init__()
        self.scale = scale
        self.num = num
        self.theta_max = np.pi / 2

    def apply(self, v):
        ### rotation vector
        angle = np.random.normal(loc=0.0, scale=self.scale * self.theta_max, size=(self.num, 1))
        angle = np.clip(angle, a_min=-self.theta_max, a_max=self.theta_max)
        # angle = np.random.uniform(-self.scale * self.theta_max, self.scale * self.theta_max, size=(self.num, 1))

        rot_axis = np.random.randn(self.num, 3)
        rot_axis /= np.linalg.norm(rot_axis, axis=-1, keepdims=True)

        rot = spatial.transform.Rotation.from_rotvec(angle * rot_axis)
        v = rot.apply(v)
        return v

    def __call__(self, v):
        return self.apply(v)


class SequentialPointcloudPatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = sum(data_source.datasets.shape_patch_count)

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class RandomPointcloudPatchSampler(torch.utils.data.sampler.Sampler):
    # Randomly get subset data from the whole dataset
    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(data_source.datasets.shape_names):
            self.total_patch_count += min(self.patches_per_shape, data_source.datasets.shape_patch_count[shape_ind])

    def __iter__(self):
        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.datasets.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class PointCloudDataset(Dataset):
    def __init__(self, root, mode=None, data_set='', data_list='', normal_init='', sparse_patch=False, with_noise=False):
        super().__init__()
        self.mode = mode
        self.data_set = data_set
        self.normal_init = normal_init
        self.sparse_patch = sparse_patch
        self.data_dir = os.path.join(root, data_set)

        self.pointclouds = []
        self.shape_names = []
        self.normals = []
        self.pidxs = []
        self.kdtrees = []
        self.shape_patch_count = []   # point number of each shape
        assert self.mode in ['train', 'val', 'test']

        ### get all shape names
        if len(data_list) > 0:
            cur_sets = []
            with open(os.path.join(root, data_set, 'list', data_list + '.txt')) as f:
                cur_sets = f.readlines()
            cur_sets = [x.strip() for x in cur_sets]
            cur_sets = list(filter(None, cur_sets))
        else:
            raise ValueError('Data list need to be given.')

        print('Current data:')
        for s in cur_sets:
            print('   ', s)
        print('Initial normal path: ', self.normal_init)

        self.load_data(cur_sets)

    def __len__(self):
        return len(self.pointclouds)

    def load_data(self, cur_sets):
        for shape_name in tqdm(cur_sets, desc='Loading data'):
            pcl = load_data(filedir=self.data_dir, filename=shape_name + '.xyz', dtype=np.float32)[:, :3]

            if len(self.normal_init) > 0:
                path_nor = os.path.join(self.normal_init, shape_name + '_normal.npy')
                if os.path.exists(path_nor):
                    nor = np.load(path_nor)
                else:
                    nor = np.loadtxt(os.path.join(self.normal_init, shape_name + '.normals'), dtype=np.float32)
                nor /= np.linalg.norm(nor, axis=-1, keepdims=True)
            elif self.mode == 'train':
                nor = load_data(filedir=self.data_dir, filename=shape_name + '.normals', dtype=np.float32)
                nor /= np.linalg.norm(nor, axis=-1, keepdims=True)
            else:
                nor = np.zeros_like(pcl)

            ### Normalization
            pcl = normalization(pcl)
            # pcl = normalization_1(pcl)

            assert pcl.shape == nor.shape
            self.pointclouds.append(pcl)
            self.normals.append(nor)
            self.shape_names.append(shape_name)

            ### KDTree construction may run out of recursions
            sys.setrecursionlimit(int(max(1000, round(pcl.shape[0]/10))))
            kdtree = spatial.cKDTree(pcl, 10)
            self.kdtrees.append(kdtree)

            if self.sparse_patch:
                pidx = load_data(filedir=self.data_dir, filename=shape_name + '.pidx', dtype=np.int32)
                self.pidxs.append(pidx)
                self.shape_patch_count.append(len(pidx))
            else:
                self.shape_patch_count.append(pcl.shape[0])

    def __getitem__(self, idx):
        ### kdtree uses a reference, not a copy of these points,
        ### so modifying the points would make the kdtree give incorrect results!
        data = {
            'pcl': self.pointclouds[idx].copy(),
            'normal': self.normals[idx],
            'kdtree': self.kdtrees[idx],
            'pidx': self.pidxs[idx] if len(self.pidxs) > 0 else None,
            'name': self.shape_names[idx],
        }

        return data


class PatchDataset(Dataset):
    def __init__(self, datasets, patch_size=1, with_trans=True, train_angle=True, sample_size=0, seed=None):
        super().__init__()
        self.datasets = datasets
        self.patch_size = patch_size
        self.trans = None
        if with_trans:
            self.trans = PCATrans()

        self.sample_size = sample_size
        self.rng_global_sample = np.random.RandomState(seed)

        self.train_angle = train_angle
        if self.train_angle:
            self.num_sample = 5000
            self.num_query = self.num_sample // 10
            self.sphere_vectors = spherical_sample(self.num_sample)
        else:
            self.add_noise = AddUniformBallNoise_Normal(scale=0.4, num=4000)  # 0.4, 4000

    def __len__(self):
        return sum(self.datasets.shape_patch_count)

    def shape_index(self, index):
        """
            Translate global (dataset-wide) point index to shape index & local (shape-wide) point index
        """
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.datasets.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset  # index in shape with ID shape_ind
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count
        return shape_ind, shape_patch_ind

    def make_patch(self, pcl, kdtree=None, nor=None, query_idx=None, patch_size=1):
        """
        Args:
            pcl: (N, 3)
            kdtree:
            nor: (N, 3)
            query_idx: (P,)
            patch_size: K
        Returns:
            pcl_pat, nor_pat: (P, K, 3)
        """
        seed_pnts = pcl[query_idx, :]
        dists, pat_idx = kdtree.query(seed_pnts, k=patch_size)  # sorted by distance (nearest first)
        dist_max = max(dists)

        pcl_pat = pcl[pat_idx, :]        # (K, 3)
        pcl_pat = pcl_pat - seed_pnts    # center
        pcl_pat = pcl_pat / dist_max     # normlize

        nor_pat = None
        if nor is not None:
            nor_pat = nor[pat_idx, :]
        return pcl_pat, nor_pat

    def make_patch_pair(self, pcl, kdtree=None, pcl_2=None, kdtree_2=None, nor=None, query_idx=None, patch_size=1, ratio=1.2):
        """
        Args:
            pcl: (N, 3)
            kdtree:
            pcl_2: (N, 3)
            kdtree_2:
            nor: (N, 3)
            query_idx: (P,)
            patch_size: K
        Returns:
            pcl_pat, nor_pat: (P, K, 3)
        """
        seed_pnts = pcl[query_idx, :]
        dists, pat_idx = kdtree.query(seed_pnts, k=patch_size)  # sorted by distance (nearest first)
        dist_max = max(dists)

        pcl_pat = pcl[pat_idx, :]        # (K, 3)
        pcl_pat = pcl_pat - seed_pnts    # center
        pcl_pat = pcl_pat / dist_max     # normlize

        dists_2, pat_idx_2 = kdtree_2.query(seed_pnts, k=patch_size*ratio)
        pcl_pat_2 = pcl_2[pat_idx_2, :]      # (K, 3)
        pcl_pat_2 = pcl_pat_2 - seed_pnts    # center
        pcl_pat_2 = pcl_pat_2 / dist_max     # normlize

        nor_pat = None
        if nor is not None:
            nor_pat = nor[pat_idx, :]
        return pcl_pat, pcl_pat_2, nor_pat

    def get_subsample(self, pts, query_idx, sample_size, pts_1=None, rng=None, fixed=False, uniform=False):
        """
            pts: (N, 3)
            query_idx: (1,)
            Warning: the query point may not be included in the output point cloud !
        """
        N_pts = pts.shape[0]
        query_point = pts[query_idx, :]

        ### if there are too much points, it is not helpful for orienting normal.
        # N_max = sample_size * 50   # TODO
        # if N_pts > N_max:
        #     point_idx = np.random.choice(N_pts, N_max, replace=False)
        #     # if query_idx not in point_idx:
        #     #     point_idx[0] = query_idx
        #     #     query_idx = 0
        #     pts = pts[point_idx, :]
        #     if pts_1 is not None:
        #         pts_1 = pts_1[point_idx, :]
        #     N_pts = N_max

        pts = pts - query_point
        dist = np.linalg.norm(pts, axis=1)
        dist_max = np.max(dist)
        pts = pts / dist_max

        if pts_1 is not None:
            pts_1 = pts_1 - query_point
            pts_1 = pts_1 / dist_max

        if N_pts >= sample_size:
            if fixed:
                rng.seed(42)

            if uniform:
                sub_ids = rng.randint(low=0, high=N_pts, size=sample_size)
            else:
                dist_normalized = dist / dist_max
                prob = 1.0 - 1.5 * dist_normalized
                prob_clipped = np.clip(prob, 0.05, 1.0)

                ids = rng.choice(N_pts, size=int(sample_size / 1.5), replace=False)
                prob_clipped[ids] = 1.0
                prob = prob_clipped / np.sum(prob_clipped)
                sub_ids = rng.choice(N_pts, size=sample_size, replace=False, p=prob)

            # Let the query point be included
            if query_idx not in sub_ids:
                sub_ids[0] = query_idx
            pts_sub = pts[sub_ids, :]
            # id_new = np.argsort(dist[sub_ids])
            # pts_sub = pts_sub[id_new, :]
        else:
            pts_shuffled = pts[:, :3]
            rng.shuffle(pts_shuffled)
            zeros_padding = np.zeros((sample_size - N_pts, 3), dtype=np.float32)
            pts_sub = np.concatenate((pts_shuffled, zeros_padding), axis=0)

        # pts_sub[0, :] = 0    # TODO
        if pts_1 is not None:
            return pts_sub, pts_1[sub_ids, :]
        return pts_sub, sub_ids

    def compute_angle_offset(self, query_vector, gt_normal, eps=1e-6):
        norm = np.linalg.norm(np.cross(query_vector, gt_normal), axis=1)
        norm[(norm < eps) & (norm > -eps)] = 0.0
        norm[norm > 1.0] = 1.0
        norm[norm < -1.0] = -1.0
        return np.arcsin(norm)

    def __getitem__(self, idx):
        """
            Returns a patch centered at the point with the given global index
            and the ground truth normal of the patch center
        """
        ### find shape that contains the point with given global index
        shape_idx, patch_idx = self.shape_index(idx)
        shape_data = self.datasets[shape_idx]

        ### get the center point
        if shape_data['pidx'] is None:
            query_idx = patch_idx
        else:
            query_idx = shape_data['pidx'][patch_idx]

        pcl_pat, normal_pat = self.make_patch(pcl=shape_data['pcl'],
                                                kdtree=shape_data['kdtree'],
                                                nor=shape_data['normal'],
                                                query_idx=query_idx,
                                                patch_size=self.patch_size,
                                            )
        normal_center = shape_data['normal'][query_idx, :]

        data = {
            'name': shape_data['name'],
            'pcl_pat': torch.from_numpy(pcl_pat).float(),
            # 'normal_pat': torch.from_numpy(normal_pat).float(),
            'normal_center': torch.from_numpy(normal_center).float(),
        }

        if self.train_angle:
            sample_ind = self.rng_global_sample.choice(self.num_sample, size=self.num_query, replace=False)
            query_vectors = self.sphere_vectors[sample_ind, :]
            angle_offsets = self.compute_angle_offset(query_vectors, normal_center)
            data['query_vectors'] = torch.from_numpy(query_vectors).float()
            data['angle_offsets'] = torch.from_numpy(angle_offsets).float()
        else:
            query_vectors = self.add_noise(normal_center)
            data['query_vectors'] = torch.from_numpy(query_vectors).float()

        if self.trans is not None:
            data = self.trans(data)
        else:
            data['pca_trans'] = torch.eye(3)
        return data


class BaseDataset(Dataset):
    def __init__(self, root, data_set, data_list, num_points=5000, num_query=10, num_knn=64, dis_k=50):
        super().__init__()
        self.num_points = num_points
        self.num_query = num_query
        self.num_knn = num_knn
        self.dis_k = dis_k
        self.num_split = 10
        self.max_point = int(3e5)
        self.data_dir = os.path.join(root, data_set)

        ### get all shape names
        if len(data_list) > 0:
            cur_sets = []
            with open(os.path.join(root, data_set, 'list', data_list + '.txt')) as f:
                cur_sets = f.readlines()
            cur_sets = [x.strip() for x in cur_sets]
            cur_sets = list(filter(None, cur_sets))
        else:
            raise ValueError('Data list need to be given.')
        for s in cur_sets:
            print('   ', s)
        self.cur_sets = cur_sets

    def load_data(self, shape_name):
        pcl = load_data(filedir=self.data_dir, filename=shape_name + '.xyz', dtype=np.float32)[:, :3]
        # pcls = []
        # n = 50  # 10 for bunny
        # boundingbox_diagonal = np.linalg.norm(pcl.max(0) - pcl.min(0))
        # for i in range(-n, n, 1):
        #     pcl1 = np.zeros_like(pcl)
        #     pcl1[:, 2] = i * 0.003  # 0.02 for bunny
        #     pcl1[:, :2] = pcl[:, :2] + np.random.normal(0.0, 1.0, size=(pcl.shape[0], 2)) * boundingbox_diagonal * 0.007
        #     pcls.append(pcl1)
        # pcl = np.concatenate(pcls, axis=0)
        # pcl[:, :2] += np.random.normal(0.0, 1.0, size=(pcl.shape[0], 2)) * 0.02
        # np.savetxt('000.xyz', pcl, fmt='%.6f')

        if os.path.exists(os.path.join(self.data_dir, shape_name + '.normals')):
            nor = load_data(filedir=self.data_dir, filename=shape_name + '.normals', dtype=np.float32)
        else:
            nor = np.zeros_like(pcl)

        pcl = normalization(pcl)
        idx = np.linalg.norm(nor, axis=-1) == 0.0
        nor /= (np.linalg.norm(nor, axis=-1, keepdims=True) + 1e-8)
        nor[idx, :] = 0.0

        self.bbox_min = np.array([np.min(pcl[:,0]), np.min(pcl[:,1]), np.min(pcl[:,2])]) - 0.05
        self.bbox_max = np.array([np.max(pcl[:,0]), np.max(pcl[:,1]), np.max(pcl[:,2])]) + 0.5

        assert pcl.shape == nor.shape
        return pcl, nor

    def process_data(self, shape_name):
        self.pt_raw = None
        self.k_idex = None
        self.pt_source = None
        self.knn_idx = None

        start_time = time.time()
        pointcloud, _ = self.load_data(shape_name)

        if pointcloud.shape[0] > self.max_point:
            print('Using sparse point cloud data.')
            # pidx = load_data(filedir=self.data_dir, filename=shape_name + '.pidx', dtype=np.int32)
            pidx = np.random.choice(pointcloud.shape[0], self.max_point, replace=False)
            pointcloud = pointcloud[pidx, :]

        if 1000000 / pointcloud.shape[0] <= 10.0:
            num_query = self.num_query
        else:
            num_query = 1000000 // pointcloud.shape[0]

        sigmas = []
        k_idex = []
        ptree = spatial.cKDTree(pointcloud)
        for p in np.array_split(pointcloud, 100, axis=0):
            d, idex = ptree.query(p, k=self.dis_k + 1)  # no self
            # d = np.clip(d, a_min=0, a_max=0.5)
            sigmas.append(d[:, -1])
            k_idex.append(idex)
        sigmas = np.concatenate(sigmas, axis=0)[:, None]
        self.k_idex = np.concatenate(k_idex, axis=0)         # (N, K)

        sample = []
        knn_idx = []
        for i in range(num_query):
            pcl_noisy = pointcloud + np.random.normal(0.0, 1.0, size=pointcloud.shape) * sigmas
            sample.append(pcl_noisy)

            for p in np.array_split(pcl_noisy, 100, axis=0):
                _, index = ptree.query(p, k=self.num_knn)
                knn_idx.append(index)
            print(i, 'Processing', shape_name)

        self.pt_source = np.concatenate(sample, axis=0)      # noisy point cloud, (num_query*N, 3)
        self.knn_idx = np.concatenate(knn_idx, axis=0)       # (num_query*N, K)
        if self.num_knn == 1:
            self.knn_idx = self.knn_idx[:, None]
        self.pt_num = self.pt_source.shape[0] - 1
        elapsed_time = time.time() - start_time              # time second

        self.pt_raw = torch.from_numpy(pointcloud).float()   # (N, 3)
        self.k_idex = torch.from_numpy(self.k_idex).long()   # (N, K1)
        print('Size:', self.pt_source.shape, '| Time: %.3fs' % elapsed_time, '\n')

    def __len__(self):
        return self.pt_source.shape[0]

    def __getitem__(self, idx):
        index_coarse = np.random.choice(self.num_split, 1)
        index_fine = np.random.choice(self.pt_num//self.num_split, self.num_points, replace=False)
        index = index_fine * self.num_split + index_coarse

        data = {
            'pcl_raw': self.pt_raw,
            # 'k_idex': self.k_idex,
            'pcl_source': torch.from_numpy(self.pt_source[index]).float(),
            'knn_idx': torch.from_numpy(self.knn_idx[index]).long(),
        }
        return data

