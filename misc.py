import os, sys
import torch
import random
import time
import logging
import numpy as np
import logging.handlers
import subprocess
from datetime import datetime


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    # if seed == 0:  # slower, more reproducible
    #     torch.backends.cudnn.benchmark = False    # default is False
    #     torch.backends.cudnn.deterministic = True
    # else:          # faster, less reproducible
    #     torch.backends.cudnn.benchmark = True    # if True, the net graph and input size should be fixed !!!
    #     torch.backends.cudnn.deterministic = False


def git_commit(logger, log_dir=None, git_name=None):
    """ Logs source code configuration
    """
    import git

    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
        git_date = datetime.fromtimestamp(repo.head.object.committed_date).strftime('%Y-%m-%d')
        git_message = repo.head.object.message
        logger.info('Source is from Commit {} ({}): {}'.format(git_sha[:8], git_date, git_message.strip()))

        # Also create diff file in the log directory
        # if log_dir is not None:
        #     with open(os.path.join(log_dir, 'compareHead.diff'), 'w') as fid:
        #         subprocess.run(['git', 'diff'], stdout=fid)

        git_name = git_name if git_name is not None else datetime.now().strftime("%y%m%d_%H%M%S")
        os.system("git add --all")
        os.system("git commit --all -m '{}'".format(git_name))
    except git.exc.InvalidGitRepositoryError:
        pass


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'), mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('Output and logs will be saved to: {}'.format(log_dir))
    return logger


def get_new_log_dir(root, prefix='', postfix=''):
    name = prefix + time.strftime("%y%m%d_%H%M%S", time.localtime()) + postfix
    log_dir = os.path.join(root, name)
    os.makedirs(log_dir)
    return log_dir, name


def log_hyperparams(writer, log_dir, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {"hp_metric": -1})
    fw = writer._get_file_writer()
    fw.add_summary(exp)
    fw.add_summary(ssi)
    fw.add_summary(sei)
    with open(os.path.join(log_dir, 'hparams.csv'), 'w') as csvf:
        csvf.write('key,value\n')
        for k, v in vars_args.items():
            csvf.write('%s,%s\n' % (k, v))


def log(args, model=None, PID=0):
    if args.logging:
        log_path, log_dir_name = get_new_log_dir(args.log_root, prefix='',
                                                postfix='_' + args.tag if args.tag is not None else '')
        sub_log_dir = os.path.join(log_path, 'log')
        os.makedirs(sub_log_dir)
        logger = get_logger(name='train(%d)(%s)' % (PID, log_dir_name), log_dir=sub_log_dir)
        # writer = torch.utils.tensorboard.SummaryWriter(sub_log_dir)
        # log_hyperparams(writer, sub_log_dir, args)
        ckpt_dir = os.path.join(log_path, 'ckpts')
        os.makedirs(ckpt_dir)
        git_commit(logger=logger, log_dir=sub_log_dir, git_name=log_dir_name)
    else:
        logger = get_logger('train', None)
        ckpt_dir = None

    ### Arguments
    logger.info('Command: {}'.format(' '.join(sys.argv)))
    arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
    logger.info('Arguments:\n' + arg_str)
    logger.info(repr(model))

    if args.logging:
        code_dir = os.path.join(log_path, 'code')
        os.makedirs(code_dir, exist_ok=True)
        os.system('cp %s %s' % ('*.py', code_dir))
        os.system('cp -r %s %s' % ('net', code_dir))

    return logger, ckpt_dir


def reset_params(module):
    for layer in module.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        reset_params(layer)


import struct
def write_pointcloud(filename, xyz, nxyz, rgb=None):
    assert xyz.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if nxyz is None:
        nxyz = np.zeros(xyz.shape).astype(np.uint8)
    assert xyz.shape == nxyz.shape, 'Input point normals should be Nx3 float array and have same size as input XYZ points'
    if rgb is None:
        rgb = np.ones(xyz.shape).astype(np.uint8)*255
    assert xyz.shape == rgb.shape, 'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename, 'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n' % xyz.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property float nx\n', 'utf-8'))
    fid.write(bytes('property float ny\n', 'utf-8'))
    fid.write(bytes('property float nz\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz.shape[0]):
        fid.write(bytearray(struct.pack("ffffffccc", xyz[i,0], xyz[i,1], xyz[i,2],
                                        nxyz[i,0], nxyz[i,1], nxyz[i,2],
                                        rgb[i,0].tostring(),
                                        rgb[i,1].tostring(),
                                        rgb[i,2].tostring())))
    fid.close()