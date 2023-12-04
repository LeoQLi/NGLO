import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mode', type=str, default='', choices=['train', 'test'])
parser.add_argument('--dataset_root', type=str, default='/data1/lq/Dataset/')
parser.add_argument('--data_set', type=str, default='PCPNet',
                    choices=['PCPNet', 'FamousShape', 'FamousShape3k', 'FamousShape5k', 'SceneNN', 'Semantic3D', 'KITTI_sub', 'Others', 'NGL_k', 'WireframePC', 'NestPC'])
parser.add_argument('--log_root', type=str, default='./log/')
parser.add_argument('--tag', type=str, default='')
### Testing
parser.add_argument('--ckpt_dirs', type=str, default='230209_002534_GVO')
parser.add_argument('--ckpt_iters', type=str, default='800')
parser.add_argument('--normal_init', type=str, default='')
args = parser.parse_args()


lr = 0.0009
encode_knn = 16
train_patch_size = 700
train_batch_size = 110

if args.mode == 'train':
    trainset_list = 'trainingset_whitenoise'
    os.system('CUDA_VISIBLE_DEVICES={} python train.py --dataset_root={} --trainset_list={} --patch_size={} --batch_size={} \
                                                    --log_root={} --encode_knn={} --lr={}'.format(
                args.gpu, args.dataset_root, trainset_list, train_patch_size, train_batch_size, args.log_root, encode_knn, lr))

elif args.mode == 'test':
    test_patch_size = train_patch_size
    test_batch_size = 400
    if args.ckpt_dirs == '':
        args.ckpt_dirs = os.path.split(os.path.abspath(os.path.dirname(os.getcwd())))[-1]

    save_normal = False
    sparse_patch = True

    testset_list = 'testset_%s' % args.data_set
    eval_list = testset_list
    if args.data_set == 'PCPNet':
        eval_list = 'testset_no_noise testset_low_noise testset_med_noise testset_high_noise \
                    testset_vardensity_striped testset_vardensity_gradient'
    elif args.data_set == 'FamousShape':
        eval_list = 'testset_noise_clean testset_noise_low testset_noise_med testset_noise_high \
                    testset_density_stripe testset_density_gradient'
    elif args.data_set == 'SceneNN':
        eval_list = 'testset_SceneNN_clean testset_SceneNN_noise'
    elif args.data_set == 'WireframePC':
        test_patch_size = 200
        sparse_patch = False

    command = 'CUDA_VISIBLE_DEVICES={} python test.py --dataset_root={} --data_set={} --log_root={} --ckpt_dirs={} --ckpt_iters={} \
                            --patch_size={} --batch_size={} --encode_knn={} --save_normal={} --sparse_patch={} --tag={} --normal_init={}'.format(
            args.gpu, args.dataset_root, args.data_set, args.log_root, args.ckpt_dirs, args.ckpt_iters,
            test_patch_size, test_batch_size, encode_knn, save_normal, sparse_patch, args.tag, args.normal_init)

    os.system('{} --testset_list={} --eval_list {}'.format(command, testset_list, eval_list))

else:
    print('The mode is unsupported!')