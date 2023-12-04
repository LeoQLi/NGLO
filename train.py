import os
import time
import argparse
import random
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from misc import seed_all, log
from net.NGLO import GVO
from datasets import PointCloudDataset, PatchDataset, RandomPointcloudPatchSampler


def parse_arguments():
    parser = argparse.ArgumentParser()
    ## Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr_gamma', type=float, default=0.2)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--scheduler_epoch', type=int, nargs='+', default=[400,600,800])
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log_root', type=str, default='')
    parser.add_argument('--tag', type=str, default='GVO')
    parser.add_argument('--nepoch', type=int, default=850)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
    ## Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--data_set', type=str, default='PCPNet')
    parser.add_argument('--trainset_list', type=str, default='trainingset_whitenoise')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--sample_size', type=int, default=0)
    parser.add_argument('--encode_knn', type=int, default=16)
    parser.add_argument('--patches_per_shape', type=int, default=1000,
                        help='The number of patches sampled from each shape in an epoch')
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    def worker_init_fn(worker_id):
        random.seed(args.seed)
        np.random.seed(args.seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='train',
            data_set=args.data_set,
            data_list=args.trainset_list,
        )
    train_set = PatchDataset(
            datasets=train_dset,
            patch_size=args.patch_size,
            sample_size=args.sample_size,
            seed=args.seed,
            train_angle=True,
        )
    train_datasampler = RandomPointcloudPatchSampler(train_set, patches_per_shape=args.patches_per_shape, seed=args.seed)
    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            sampler=train_datasampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=g,
        )

    return train_dataloader, train_datasampler


def train(epoch):
    for batch_idx, batch in enumerate(train_dataloader, 0):
        pcl_pat = batch['pcl_pat'].to(_device)
        query_vectors = batch['query_vectors'].to(_device)            # (B, M, 3)
        angle_offsets = batch['angle_offsets'].to(_device)            # (B, M)
        normal_center = batch['normal_center'].to(_device).squeeze()  # (B, 3)

        ### Reset grad and model state
        model.train()
        optimizer.zero_grad()

        ### Forward
        pred_point, weights = model(pcl_pat, query_vectors=query_vectors)
        loss, loss_tuple = model.get_loss(q_target=normal_center,
                                        angle_pred=pred_point, angle_gt=angle_offsets,
                                        pred_weights=weights, pcl_in=pcl_pat,
                                    )

        ### Backward and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        ### Logging
        if batch_idx % 10 == 0:
            ss = ''
            for l in loss_tuple:
                ss += '%.5f+' % l.item()
            logger.info('[Train] [%03d: %03d/%03d] | Loss: %.6f(%s) | Grad: %.6f' % (
                        epoch, batch_idx, train_num_batch-1, loss.item(), ss[:-1], orig_grad_norm)
                    )
    return 1


def scheduler_fun():
    pre_lr = optimizer.param_groups[0]['lr']
    current_lr = pre_lr * args.lr_gamma
    if current_lr < args.lr_min:
        current_lr = args.lr_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    logger.info('Update learning rate: %f => %f \n' % (pre_lr, current_lr))


### Arguments
args = parse_arguments()
_device = torch.device('cuda:%d' % args.gpu)
seed_all(args.seed)
PID = os.getpid()

### Model
print('Building model ...')
model = GVO(num_pat=args.patch_size, encode_knn=args.encode_knn).to(_device)

### Datasets and loaders
print('Loading datasets ...')
train_dataloader, train_datasampler = get_data_loaders(args)
train_num_batch = len(train_dataloader)

### Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)

### Log
logger, ckpt_dir = log(args, model, PID)
logger.info('training set: %d patches (in %d batches)' % (len(train_datasampler), len(train_dataloader)))

if __name__ == '__main__':
    logger.info('Start training ...')
    try:
        for epoch in range(1, args.nepoch+1):
            logger.info('### Epoch %d ###' % epoch)

            start_time = time.time()
            train(epoch)
            end_time = time.time()
            logger.info('Time cost: %.1f s \n' % (end_time-start_time))

            if epoch in args.scheduler_epoch:
                scheduler_fun()

            if epoch % args.interval == 0 or epoch == args.nepoch:
                if args.logging:
                    model_filename = os.path.join(ckpt_dir, 'ckpt_%d.pt' % epoch)
                    torch.save(model.state_dict(), model_filename)

    except KeyboardInterrupt:
        logger.info('Terminating ...')
