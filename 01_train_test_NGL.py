import os, sys
import argparse
import time
import math
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from misc import log, seed_all, get_logger
from net.NGLO import NGL
from datasets import BaseDataset
from mesh import extract_mesh


### Arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    ## Training
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_gamma', type=float, default=0.2)
    parser.add_argument('--scheduler_step', type=int, default=20000)
    parser.add_argument('--max_iter', type=int, default=40000)
    parser.add_argument('--save_inter', type=int, default=20000)
    ## Dataset and loader
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--dataset_root', type=str, default='/data1/lq/Dataset/')
    parser.add_argument('--data_set', type=str, default='PCPNet')
    parser.add_argument('--testset_list', type=str, default='')
    parser.add_argument('--num_points', type=int, default=5000)
    parser.add_argument('--num_query', type=int, default=10)
    parser.add_argument('--num_knn', type=int, default=64)
    parser.add_argument('--dis_k', type=int, default=50)
    ## Test
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--ckpt_iter', type=int, default=None)
    parser.add_argument('--save_normal', type=eval, default=True, choices=[True, False])
    parser.add_argument('--save_mesh', type=eval, default=False, choices=[True, False])
    parser.add_argument('--mesh_far', type=float, default=-1.0)
    args = parser.parse_args()
    return args


def train():
    ### Dataset
    train_set = BaseDataset(root=args.dataset_root,
                            data_set=args.data_set,
                            data_list=args.testset_list,
                            num_points=args.num_points,
                            num_query=args.num_query,
                            num_knn=args.num_knn,
                            dis_k=args.dis_k,
                        )
    dataloader = torch.utils.data.DataLoader(
                            train_set,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                        )

    with_log = True
    num_shapes = len(train_set.cur_sets)
    for shape_idx, shape_name in enumerate(train_set.cur_sets):
        ### Model
        print('Building model ...')
        model_ngl = NGL().to(_device).train()
        optimizer = optim.Adam(model_ngl.parameters(), lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.lr_gamma)

        train_set.process_data(shape_name)
        iter_dataloader = iter(dataloader)

        ### Log
        if with_log:
            logger, ckpt_dir = log(args, model_ngl, PID)
            with_log = False

        logger.info('Training: ' + shape_name)
        start_time = time.time()
        for iter_i in range(1, args.max_iter+1):
            data = iter_dataloader.next()
            pcl_raw = data['pcl_raw'].to(_device)
            pcl_source = data['pcl_source'].to(_device)
            knn_idx = data['knn_idx'].to(_device)

            ### Reset gradient and model state
            model_ngl.train()
            optimizer.zero_grad()

            grad_norm = model_ngl(pcl_source=pcl_source)
            loss = model_ngl.get_loss(pcl_raw=pcl_raw, pcl_source=pcl_source, knn_idx=knn_idx)

            ### Backward and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()

            if iter_i % (args.save_inter//10) == 0:
                logger.info('shape:%d/%d, iter:%d/%d, loss=%.6f, lr=%.8f' % (
                            shape_idx+1, num_shapes, iter_i, args.max_iter, loss, optimizer.param_groups[0]['lr']))

            if iter_i % args.save_inter == 0 or iter_i == args.max_iter:
                model_filename = os.path.join(ckpt_dir, shape_name + '_%d.pt' % iter_i)
                torch.save(model_ngl.state_dict(), model_filename)
                logger.info('Save model: ' + model_filename)
                # pc_nor = torch.cat([pcl_source, grad_norm], dim=-1)[0].cpu().detach().numpy()
                # np.savetxt(model_filename[:-3] + '.txt', pc_nor, fmt='%.6f')

        elapsed_time = time.time() - start_time
        logger.info('Time: %.2f sec\n' % elapsed_time)

    return 1


def test():
    ### Dataset
    test_set = BaseDataset(root=args.dataset_root,
                            data_set=args.data_set,
                            data_list=args.testset_list,
                        )

    ### Model
    print('Building model ...')
    model_ngl = NGL().to(_device).eval()

    ### Log
    output_dir = os.path.join(args.log_root, args.ckpt_dir, 'test_%s' % args.ckpt_iter)
    save_dir = os.path.join(output_dir, 'pred_normal')
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger('test(%d)(%s-%s)' % (PID, args.ckpt_dir, args.ckpt_iter), output_dir)
    logger.info('Command: {}'.format(' '.join(sys.argv)))

    trainable_num = sum(p.numel() for p in model_ngl.parameters() if p.requires_grad)
    logger.info('Num_params_trainable: %d' % trainable_num)

    max_n = int(2e5)
    list_bad = {}
    list_rms = []
    list_rms_o = []
    list_p90 = []
    start_time = time.time()
    for shape_idx, shape_name in enumerate(test_set.cur_sets):
        ### load the trained model
        ckpt_path = os.path.join(args.log_root, args.ckpt_dir, 'ckpts/%s_%s.pt' % (shape_name, args.ckpt_iter))
        if not os.path.exists(ckpt_path):
            print('File not exist:', ckpt_path)
            continue
        model_ngl.load_state_dict(torch.load(ckpt_path, map_location=_device), strict=False)

        ### load a point cloud
        pcl_raw, nor_gt = test_set.load_data(shape_name)         # (N, 3)
        num_point = pcl_raw.shape[0]
        rand_idxs = np.random.choice(num_point, num_point, replace=False)
        pcl = pcl_raw[rand_idxs, :]

        if num_point <= max_n:
            pcl_source = torch.from_numpy(pcl).float().to(_device)
            with torch.no_grad():
                grad_norm = model_ngl(pcl_source=pcl_source)
                grad_norm = grad_norm.cpu().detach().numpy()
        else:
            k = math.ceil(num_point / max_n)
            remainder = int(max_n * k % num_point)
            print(num_point, k, remainder)
            pcl_new = np.concatenate((pcl, pcl[:remainder, :]), axis=0)
            pcl_source = torch.from_numpy(pcl_new).float()   # (max_n*k, 3)
            grad_norm = np.zeros_like(pcl_new)
            with torch.no_grad():
                for i in range(k):
                    grad_norm_s = model_ngl(pcl_source=pcl_source[max_n*i:max_n*(i+1), :].to(_device))
                    grad_norm[max_n*i:max_n*(i+1), :] = grad_norm_s.cpu().detach().numpy()
            grad_norm = grad_norm[:max_n*k-remainder, :]

        pred_norm = np.zeros_like(grad_norm)
        pred_norm[rand_idxs, :] = grad_norm
        pred_norm[np.linalg.norm(pred_norm, axis=-1) == 0.0, :] = 1.0
        pred_norm /= np.linalg.norm(pred_norm, axis=-1, keepdims=True)

        assert pcl_raw.shape == pred_norm.shape
        if args.save_normal:
            path_save = os.path.join(save_dir, shape_name)
            # pc_nor = np.concatenate([pcl_raw, pred_norm], axis=-1)
            # np.savetxt(path_save + '.txt', pc_nor, fmt='%.6f')
            np.save(path_save + '_normal.npy', pred_norm)
            logger.info('Save file: ' + path_save)

        nn = np.sum(np.multiply(-1 * nor_gt, pred_norm), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1
        ang = np.rad2deg(np.arccos(np.abs(nn)))
        rms = np.sqrt(np.mean(np.square(ang)))

        ang_o = np.rad2deg(np.arccos(nn))
        ids = ang_o < 90.0
        p90 = sum(ids) / pred_norm.shape[0] * 100

        ### if more than half of points have wrong orientation, then flip all normals
        if p90 < 50.0:
            nn = np.sum(np.multiply(nor_gt, pred_norm), axis=1)
            nn[nn > 1] = 1
            nn[nn < -1] = -1
            ang_o = np.rad2deg(np.arccos(nn))
            ids = ang_o < 90.0
            p90 = sum(ids) / pred_norm.shape[0] * 100

        rms_o = np.sqrt(np.mean(np.square(ang_o)))
        list_rms.append(rms)
        list_rms_o.append(rms_o)
        list_p90.append(p90)
        if (np.mean(p90) < 90.0):
            list_bad[shape_name] = p90
        logger.info('RMSE_U: %.3f, RMSE_O: %.3f, Correct orientation: %.3f %% (%s)' % (rms, rms_o, p90, shape_name))

        if args.save_mesh:
            mesh_dir = os.path.join(output_dir, 'recon_mesh')
            os.makedirs(mesh_dir, exist_ok=True)
            mesh = extract_mesh(model_ngl.net.forward, bbox_min=test_set.bbox_min, bbox_max=test_set.bbox_max,
                                points_gt=pcl_raw, mesh_far=args.mesh_far)
            mesh.export(os.path.join(mesh_dir, '%s.obj' % shape_name))

    elapsed_time = time.time() - start_time
    if len(list_p90) > 0:
        logger.info('Time: %.2f sec\n' % elapsed_time)
        logger.info('Average || RMSE_U: %.3f, RMSE_O: %.3f, Correct orientation: %.3f %%' % (np.mean(list_rms), np.mean(list_rms_o), np.mean(list_p90)))
        ss = ''
        for k, v in list_bad.items():
            ss += '%s: %.3f %%\n' % (k, v)
        logger.info('Bad results in %d shapes: \n%s' % (len(list_p90), ss))

    return 1



if __name__ == '__main__':
    args = parse_arguments()
    args.tag = args.data_set

    if len(args.testset_list) == 0:
        args.testset_list = 'testset_' + args.data_set

    if args.data_set in ['FamousShape3k', 'FamousShape5k']:
        args.max_iter = 20000
        args.save_inter = 10000
        args.num_points = 1000
        args.num_knn = 32
        args.dis_k = 5
    elif args.data_set == 'WireframePC':
        args.max_iter = 5000
        args.save_inter = 2500
        args.num_points = 300
        args.num_knn = 5
        args.dis_k = 3
    elif args.data_set == 'NestPC':
        args.num_knn = 32
        args.dis_k = 20
    elif args.data_set in ['SceneNN', 'Semantic3D', 'KITTI_sub', 'Others']:
        args.lr = 0.00001
        args.dis_k = 64

    _device = torch.device('cuda:%d' % args.gpu)
    PID = os.getpid()
    seed_all(args.seed)

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        print('The mode is unsupported!')