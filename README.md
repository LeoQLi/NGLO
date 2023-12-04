# Neural Gradient Learning and Optimization for Oriented Point Normal Estimation (SIGGRAPH Asia 2023)

### **[Project](https://leoqli.github.io/NGLO/) | [arXiv](https://arxiv.org/abs/2309.09211)**

We propose Neural Gradient Learning (NGL), a deep learning approach to learn gradient vectors with consistent orientation from 3D point clouds for normal estimation. It has excellent gradient approximation properties for the underlying geometry of the data. We utilize a simple neural network to parameterize the objective function to produce gradients at points using a global implicit representation. However, the derived gradients usually drift away from the ground-truth oriented normals due to the lack of local detail descriptions. Therefore, we introduce Gradient Vector Optimization (GVO) to learn an angular distance field based on local plane geometry to refine the coarse gradient vectors. Finally, we formulate our method with a two-phase pipeline of coarse estimation followed by refinement. Moreover, we integrate two weighting functions, i.e., anisotropic kernel and inlier score, into the optimization to improve the robust and detail-preserving performance. Our method efficiently conducts global gradient approximation while achieving better accuracy and generalization ability of local feature description. This leads to a state-of-the-art normal estimator that is robust to noise, outliers and point density variations. Extensive evaluations show that our method outperforms previous works in both unoriented and oriented normal estimation on widely used benchmarks.

## Requirements

The code is implemented in the following environment settings:
- Ubuntu 16.04
- CUDA 10.1
- Python 3.8
- Pytorch 1.8
- Pytorch3d 0.6
- Numpy 1.23
- Scipy 1.6

We train and test our code on an NVIDIA 2080 Ti GPU.

## Dataset
The datasets used in our paper can be downloaded from [Here](https://drive.google.com/drive/folders/1eNpDh5ivE7Ap1HkqCMbRZpVKMQB1TQ6H?usp=share_link).
Unzip them to a folder `***/dataset/` and set the path value of `dataset_root` in `01_train_test_NGL.py` and `02_run_GVO.py`.
The dataset is organized as follows:
```
│dataset/
├──PCPNet/
│  ├── list
│      ├── ***.txt
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
├──FamousShape/
│  ├── list
│      ├── ***.txt
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
```

## Train
- Neural Gradient Learning (NGL)
```
python 01_train_test_NGL.py --mode=train --gpu=0 --data_set=***
```
- Gradient Vector Optimization (GVO)
```
python 02_run_GVO.py --mode=train --gpu=0
```
For NGL, you need to set `data_set` according to different datasets.
By default, we only use the training set of the PCPNet dataset to train our GVO.
The trained network models will be save in `./log/***/`.

## Test
Our pre-trained models can be downloaded from [Here](https://drive.google.com/drive/folders/1tJq8HiEIUTfsvWIZe03Q1B37-IM__hkZ?usp=sharing).

To test on the PCPNet dataset using the provided models, simply run:
- Neural Gradient Learning (NGL)
```
python 01_train_test_NGL.py --mode=test --gpu=0 --data_set=PCPNet --ckpt_dir=230208_021652_PCPNet --ckpt_iter=40000
```
- Gradient Vector Optimization (GVO)
```
python 02_run_GVO.py --mode=test --gpu=0 --data_set=PCPNet --ckpt_dirs=230209_002534_GVO --ckpt_iters=800 --normal_init=./log/230208_021652_PCPNet/test_40000/pred_normal
```
The predicted normals and evaluation results will be saved in `./log/230209_002534_GVO/results_PCPNet_230208_021652/ckpt_800/`.

Maybe you want to change two variables in `02_run_GVO.py`:
```
save_normal  = True    # to save the estimated point normals
sparse_patch = True    # to test on sparse point clouds based on '.pidx' files
```

## Citation
If you find our work useful in your research, please cite our paper:

    @inproceedings{li2023neural,
      title={Neural Gradient Learning and Optimization for Oriented Point Normal Estimation},
      author={Li, Qing and Feng, Huifang and Shi, Kanle and Fang, Yi and Liu, Yu-Shen and Han, Zhizhong},
      booktitle={SIGGRAPH Asia 2023 Conference Papers},
      year={2023},
      month={December}
    }

