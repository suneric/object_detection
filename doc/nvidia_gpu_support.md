## nvidia gpu support

## Install nvidia driver
- check the system driver requirement
```
ubuntu-drivers devices
```
RTX3070 requires nvidia-driver-460 as recommended version
- Install nvidia-driver-460 and cuda 11.0 (requires driver version >= 450)

## Install cuda 11.0
- follow the instruction of [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu) to install cuda and cudnn
- check gpu enabled and [use gpu](https://www.tensorflow.org/guide/gpu) once tensorflow is installed
```
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(len(gpus))
```

## install cudatoolkit 11.0, cudnn 8 and tensorflow 2.4 in anaconda
- install cudatoolkit=11.0 with conda (latest shuld be 11.0.221)
- pip install tensorflow (latest should be 2.4)
- download cudnn 8 and unzip, copy the include files and lib64 file to your anaconda environment (include folder and lib folder)

## References
- [Tensorflow GPU support](https://www.tensorflow.org/install/gpu)
- [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
- [Install TF2.4 with CUDA 11 and CUDNN 8](https://medium.com/analytics-vidhya/install-tensorflow-gpu-2-4-0-with-cuda-11-0-and-cudnn-8-using-anaconda-8c6472c9653f)
- [Deep Learning with RTX3090(CUDA, cuDNN, Tensorflow, Keeras, PyTorch)](https://medium.com/@dun.chwong/the-simple-guide-deep-learning-with-rtx-3090-cuda-cudnn-tensorflow-keras-pytorch-e88a2a8249bc)
- [How to install latest cuda](https://stackoverflow.com/questions/55256671/how-to-install-latest-cudnn-to-conda)
- [CUDA Application compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#cuda-application-compatibility)
