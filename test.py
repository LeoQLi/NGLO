import os, sys, re
import shutil
import time
import argparse
import torch
import numpy as np

from net.NGLO import GVO
from misc import get_logger, seed_all
from datasets import PointCloudDataset, PatchDataset, SequentialPointcloudPatchSampler, load_data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--data_set', type=str, default='')
    parser.add_argument('--log_root', type=str, default='')
    parser.add_argument('--ckpt_dirs', type=str, default='', help="multiple files separated by comma")
    parser.add_argument('--ckpt_iters', type=str, default='', help="multiple files separated by comma")
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--testset_list', type=str, default='')
    parser.add_argument('--eval_list', type=str, nargs='*',
                        help='list of files containing sets of point cloud names for evaluation')
    parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--sample_size', type=int, default=0)
    parser.add_argument('--encode_knn', type=int, default=16)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--normal_init', type=str, default='')
    parser.add_argument('--sparse_patch', type=eval, default=True, choices=[True, False],
                        help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--save_normal', type=eval, default=False, choices=[True, False])
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    test_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='test',
            data_set=args.data_set,
            data_list=args.testset_list,
            normal_init=args.normal_init,
            sparse_patch=args.sparse_patch,
        )
    test_set = PatchDataset(
            datasets=test_dset,
            patch_size=args.patch_size,
            sample_size=args.sample_size,
            seed=args.seed,
            train_angle=False,
        )
    test_dataloader = torch.utils.data.DataLoader(
            test_set,
            sampler=SequentialPointcloudPatchSampler(test_set),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    return test_dset, test_dataloader


def normal_RMSE(normal_gts, normal_preds, eval_file='log.txt'):
    """
        Compute normal root-mean-square error (RMSE)
    """
    def l2_norm(v):
        norm_v = np.sqrt(np.sum(np.square(v), axis=1))
        return norm_v

    log_file = open(eval_file, 'w')
    def log_string(out_str):
        log_file.write(out_str+'\n')
        log_file.flush()
        # print(out_str)

    rms = []
    rms_o = []
    pgp30 = []
    pgp25 = []
    pgp20 = []
    pgp15 = []
    pgp10 = []
    pgp5 = []
    pgp90_o = []
    pgp_alpha = []
    pgp_alpha_o = []

    for i in range(len(normal_gts)):
        normal_gt = normal_gts[i]
        normal_pred = normal_preds[i]

        normal_gt_norm = l2_norm(normal_gt)
        normal_results_norm = l2_norm(normal_pred)
        normal_pred = np.divide(normal_pred, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
        normal_gt = np.divide(normal_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

        ### Unoriented RMSE
        ####################################################################
        nn = np.sum(np.multiply(normal_gt, normal_pred), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1

        ang = np.rad2deg(np.arccos(np.abs(nn)))

        ### Unoriented error metrics, portion of good points (PGP)
        rms.append(np.sqrt(np.mean(np.square(ang))))
        pgp30_shape = sum([j < 30.0 for j in ang]) / float(len(ang))
        pgp25_shape = sum([j < 25.0 for j in ang]) / float(len(ang))
        pgp20_shape = sum([j < 20.0 for j in ang]) / float(len(ang))
        pgp15_shape = sum([j < 15.0 for j in ang]) / float(len(ang))
        pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))
        pgp5_shape  = sum([j < 5.0 for j in ang])  / float(len(ang))
        pgp30.append(pgp30_shape)
        pgp25.append(pgp25_shape)
        pgp20.append(pgp20_shape)
        pgp15.append(pgp15_shape)
        pgp10.append(pgp10_shape)
        pgp5.append(pgp5_shape)

        pgp_alpha_shape = []
        for alpha in range(30):
            pgp_alpha_shape.append(sum([j < alpha for j in ang]) / float(len(ang)))
        pgp_alpha.append(pgp_alpha_shape)

        ### Oriented RMSE
        ####################################################################
        ang_o = np.rad2deg(np.arccos(nn))   # angle error in degree
        ids = ang_o < 90.0
        p90 = sum(ids) / normal_pred.shape[0]

        ### if more than half of points have wrong orientation, then flip all normals
        if p90 < 0.5:
            nn = np.sum(np.multiply(normal_gt, -1 * normal_pred), axis=1)
            nn[nn > 1] = 1
            nn[nn < -1] = -1
            ang_o = np.rad2deg(np.arccos(nn))    # angle error in degree
            ids = ang_o < 90.0
            p90 = sum(ids) / normal_pred.shape[0]

        pgp90_o.append(p90 * 100)
        rms_o.append(np.sqrt(np.mean(np.square(ang_o))))

        ### for drawing curve
        pgp_alpha_shape_o = []
        for alpha in range(90):
            pgp_alpha_shape_o.append(sum([j < alpha for j in ang_o]) / float(len(ang_o)))
        pgp_alpha_o.append(pgp_alpha_shape_o)

        # diff = np.arccos(nn)
        # diff_inv = np.arccos(-nn)
        # unoriented_normals = normal_pred
        # unoriented_normals[diff_inv < diff, :] = -normal_pred[diff_inv < diff, :]

    avg_rms   = np.mean(rms)
    avg_rms_o = np.mean(rms_o)
    avg_pgp30 = np.mean(pgp30)
    avg_pgp25 = np.mean(pgp25)
    avg_pgp20 = np.mean(pgp20)
    avg_pgp15 = np.mean(pgp15)
    avg_pgp10 = np.mean(pgp10)
    avg_pgp5  = np.mean(pgp5)
    avg_pgp_alpha = np.mean(np.array(pgp_alpha), axis=0)
    avg_pgp90_o = np.mean(pgp90_o)
    avg_pgp_alpha_o = np.mean(np.array(pgp_alpha_o), axis=0)

    log_string('RMS per shape: ' + str(rms))
    log_string('RMS_O per shape: ' + str(rms_o))
    log_string('RMS not oriented (shape average): ' + str(avg_rms))
    log_string('RMS oriented (shape average): ' + str(avg_rms_o))
    log_string('PGP30 per shape: ' + str(pgp30))
    log_string('PGP25 per shape: ' + str(pgp25))
    log_string('PGP20 per shape: ' + str(pgp20))
    log_string('PGP15 per shape: ' + str(pgp15))
    log_string('PGP10 per shape: ' + str(pgp10))
    log_string('PGP5 per shape: ' + str(pgp5))
    log_string('PGP30 average: ' + str(avg_pgp30))
    log_string('PGP25 average: ' + str(avg_pgp25))
    log_string('PGP20 average: ' + str(avg_pgp20))
    log_string('PGP15 average: ' + str(avg_pgp15))
    log_string('PGP10 average: ' + str(avg_pgp10))
    log_string('PGP5 average: ' + str(avg_pgp5))
    log_string('PGP alpha average: ' + str(avg_pgp_alpha))
    log_string('PGP_O alpha average: ' + str(avg_pgp_alpha_o))
    log_string('PGP90_O per shape (%): ' + str(pgp90_o))
    log_string('PGP90_O average (%): ' + str(avg_pgp90_o))
    log_file.close()

    return avg_rms, avg_rms_o


def eval_normal(eval_list, normal_gt_path, normal_pred_path, output_dir):
    print('\n  Evaluation ...')
    eval_summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(eval_summary_dir, exist_ok=True)

    all_avg_rms = []
    all_avg_rms_o = []
    for cur_list in eval_list:
        print("\n***************** " + cur_list + " *****************")
        print("Result path: " + normal_pred_path)

        ### get all shape names in the list
        shape_names = []
        normal_gt_filenames = os.path.join(normal_gt_path, 'list', cur_list + '.txt')
        with open(normal_gt_filenames) as f:
            shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))

        ### load all data of the list
        normal_gts = []
        normal_preds = []
        for shape in shape_names:
            print(shape)
            normal_pred = np.load(os.path.join(normal_pred_path, shape + '_normal.npy'))                  # (n, 3)
            points_idx = load_data(filedir=normal_gt_path, filename=shape + '.pidx', dtype=np.int32)      # (n,)
            normal_gt = load_data(filedir=normal_gt_path, filename=shape + '.normals', dtype=np.float32)  # (N, 3)
            normal_gt = normal_gt[points_idx, :]
            if normal_pred.shape[0] > normal_gt.shape[0]:
                normal_pred = normal_pred[points_idx, :]
            normal_gts.append(normal_gt)
            normal_preds.append(normal_pred)

        ### compute RMSE per-list
        avg_rms, avg_rms_o = normal_RMSE(normal_gts=normal_gts,
                            normal_preds=normal_preds,
                            eval_file=os.path.join(eval_summary_dir, cur_list + '_evaluation_results.txt'))
        all_avg_rms.append(avg_rms)
        all_avg_rms_o.append(avg_rms_o)

        print('### RMSE: %f' % avg_rms)
        print('### RMSE_Ori: %f' % avg_rms_o)

    s = '\n{} \nAll RMS not oriented (shape average): {} | Mean: {}'.format(
                normal_pred_path, str(all_avg_rms), np.mean(all_avg_rms))
    print(s)

    s = '\n{} \nAll RMS oriented (shape average): {} | Mean: {}'.format(
                normal_pred_path, str(all_avg_rms_o), np.mean(all_avg_rms_o))
    print(s)

    return all_avg_rms, all_avg_rms_o


def infer_iter(regressor, pcl_pat, query_vectors):

    def average_normals(refined_normal):
        first_normal = refined_normal[:, 0, :].unsqueeze(-1)
        sign = torch.sign(torch.matmul(refined_normal, first_normal))
        signed_normal = refined_normal * sign
        n_pred = signed_normal.mean(dim=-2)
        return torch.nn.functional.normalize(n_pred, dim=-1)

    query_vectors = torch.autograd.Variable(query_vectors, requires_grad=True)
    optimizer = torch.optim.Adam([query_vectors], lr=0.01)

    iter_num = 3
    loss_fun = torch.nn.L1Loss()
    for i in range(iter_num):
        optimizer.zero_grad()

        angle_pred = regressor(pcl_pat, query_vectors=query_vectors, mode_test=True)  # (B, M)

        loss = loss_fun(angle_pred, torch.zeros_like(angle_pred))
        loss.backward()
        optimizer.step()

    n_est = average_normals(query_vectors.detach())

    return n_est    # (B, 3)


def infer_one(regressor, pcl_pat, query_vectors):

    def choose_min_query_vector(angle_pred, query_vectors, n=1):
        angle_pred = abs(angle_pred)
        sorted_angle, ind = torch.sort(angle_pred, dim=-1)   # in ascending order
        arrange = torch.arange(0, angle_pred.shape[0]).unsqueeze(1).repeat(1, angle_pred.shape[1])  # (B, M)
        query_vectors = query_vectors[arrange, ind]          # (B, M, 3)
        return query_vectors[:, 0:n, :], sorted_angle[:, 0:n]

    angle_pred = regressor(pcl_pat, query_vectors=query_vectors, mode_test=True)    # (B, M)

    n_est, n_ang = choose_min_query_vector(angle_pred, query_vectors, 1)            # (B, 1, 3)

    return n_est.squeeze(1)


def test(ckpt_dir, ckpt_iter):
    ### Input/Output
    time_str = re.findall(r'\d+\_+\d+', args.normal_init)   # ***_***
    ckpt_path = os.path.join(args.log_root, ckpt_dir, 'ckpts/ckpt_%s.pt' % ckpt_iter)
    if len(time_str) > 0:
        assert len(time_str) == 1
        output_dir = os.path.join(args.log_root, ckpt_dir, 'results_%s_%s/ckpt_%s' % (args.data_set, time_str[0], ckpt_iter))
    else:
        output_dir = os.path.join(args.log_root, ckpt_dir, 'results_%s/ckpt_%s' % (args.data_set, ckpt_iter))
    if args.tag is not None and len(args.tag) != 0:
        output_dir += '_' + args.tag
    if not os.path.exists(ckpt_path):
        print('ERROR path: %s' % ckpt_path)
        return False, False

    file_save_dir = os.path.join(output_dir, 'pred_normal')
    os.makedirs(file_save_dir, exist_ok=True)

    logger = get_logger('test(%d)(%s-%s)' % (PID, ckpt_dir, ckpt_iter), output_dir)
    logger.info('Command: {}'.format(' '.join(sys.argv)))

    ### Model
    logger.info('Loading model: %s' % ckpt_path)
    model = GVO(num_pat=args.patch_size, encode_knn=args.encode_knn).to(_device)

    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # trainable_num = sum([np.prod(p.size()) for p in model_parameters])
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of trainable parameters: %d' % trainable_num)

    model.load_state_dict(torch.load(ckpt_path, map_location=_device))
    model.eval()

    num_batch = len(test_dataloader)
    shape_ind = 0
    shape_patch_offset = 0
    shape_num = len(test_dset.shape_names)
    shape_patch_count = test_dset.shape_patch_count[shape_ind]
    normal_prop = torch.zeros([shape_patch_count, 3])

    total_time = 0
    for batchind, batch in enumerate(test_dataloader, 0):
        pcl_pat = batch['pcl_pat'].to(_device)                        # (B, N, 3)
        data_trans = batch['pca_trans'].to(_device)                   # (B, 3, 3)

        if len(args.normal_init) == 0:
            query_num = 10000
            query_vectors = torch.randn((args.batch_size, query_num, 3), dtype=torch.float32)
            query_vectors = torch.nn.functional.normalize(query_vectors, dim=-1).to(_device)
        else:
            query_vectors = batch['query_vectors'].to(_device)        # (B, M, 3)

        start_time = time.time()
        with torch.no_grad():
            n_est = infer_one(regressor=model, pcl_pat=pcl_pat, query_vectors=query_vectors)
        # n_est = infer_iter(regressor=model, pcl_pat=pcl_pat, query_vectors=query_vectors)
        end_time = time.time()
        elapsed_time = end_time - start_time    # second
        total_time += elapsed_time

        if batchind % 5 == 0:
            bs = pcl_pat.size()[0]
            logger.info('[%d/%d] %s: time per patch: %.4f ms' % (
                        batchind, num_batch-1, test_dset.shape_names[shape_ind], 1000*elapsed_time / bs))

        if data_trans is not None:
            ### inverse pca rotation (back to world space)
            n_est = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)

        ### Save the estimated normals to file
        batch_offset = 0
        while batch_offset < n_est.shape[0] and shape_ind + 1 <= shape_num:
            shape_patches_remaining = shape_patch_count - shape_patch_offset
            batch_patches_remaining = n_est.shape[0] - batch_offset
            patches_remaining = min(shape_patches_remaining, batch_patches_remaining)

            ### append estimated patch properties batch to properties for the current shape on the CPU
            normal_prop[shape_patch_offset:shape_patch_offset + patches_remaining, :] = \
                n_est[batch_offset:batch_offset + patches_remaining, :]

            batch_offset += patches_remaining
            shape_patch_offset += patches_remaining

            if shape_patches_remaining <= batch_patches_remaining:
                normals_to_write = normal_prop.cpu().numpy()

                save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '_normal.npy') # faster reading speed
                np.save(save_path, normals_to_write)
                # if args.save_normal:
                #     save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '.normals')
                #     np.savetxt(save_path, normals_to_write, fmt='%.6f')

                logger.info('Save normal: {}'.format(save_path))
                logger.info('Total Time: %.2f sec, Shape Num: %d / %d \n' % (total_time, shape_ind+1, shape_num))

                sys.stdout.flush()
                shape_patch_offset = 0
                shape_ind += 1
                if shape_ind < shape_num:
                    shape_patch_count = test_dset.shape_patch_count[shape_ind]
                    normal_prop = torch.zeros([shape_patch_count, 3])

    logger.info('Total Time: %.2f sec, Shape Num: %d' % (total_time, shape_num))
    return output_dir, file_save_dir



if __name__ == '__main__':
    ### Arguments
    args = parse_arguments()

    arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
    print('Arguments:\n %s\n' % arg_str)

    _device = torch.device('cuda:%d' % args.gpu)
    PID = os.getpid()
    seed_all(args.seed)

    ### Datasets and loaders
    test_dset, test_dataloader = get_data_loaders(args)

    ckpt_dirs = args.ckpt_dirs.split(',')
    ckpt_iters = args.ckpt_iters.split(',')

    for ckpt_dir in ckpt_dirs:
        eval_dict = ''
        sum_file = 'eval_' + args.data_set + ('_'+args.tag if len(args.tag) != 0 else '')
        log_file_sum = open(os.path.join(args.log_root, ckpt_dir, sum_file+'.txt'), 'a')
        log_file_sum.write('\n====== %s ======\n' % args.eval_list)

        for ckpt_iter in ckpt_iters:
            output_dir, file_save_dir = test(ckpt_dir=ckpt_dir, ckpt_iter=ckpt_iter)

            if args.data_set in ['Semantic3D', 'WireframePC', 'KITTI_sub', 'Others']:
                continue

            all_avg_rms, all_avg_rms_o = eval_normal(eval_list=args.eval_list,
                                            normal_gt_path=os.path.join(args.dataset_root, args.data_set),
                                            normal_pred_path=file_save_dir,
                                            output_dir=output_dir,
                                        )
            s = '%s: %s | Mean: %f \t|| %s | Mean: %f\n' % (ckpt_iter, str(all_avg_rms), np.mean(all_avg_rms),
                                                                    str(all_avg_rms_o), np.mean(all_avg_rms_o))
            log_file_sum.write(s)
            log_file_sum.flush()
            eval_dict += s

            ### delete the output point cloud normals
            if not args.save_normal:
                shutil.rmtree(file_save_dir)

        log_file_sum.close()
        s = '\nAll RMS unoriented and oriented (shape average): \n{}\n'.format(eval_dict)
        print(s)


