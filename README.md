<div align="center">
# Project2:FaceSwap

<img src="./original_video.gif" alt="demo_vid_0" style="zoom:30%;" />

<img src="./swap_video.gif" alt="demo_vid_0" style="zoom:30%;" />




## Abstract
>

![overview](assets/a.png)

For a quick play around, you can check out a version of FaceDancer hosted on [Hugging Face](https://huggingface.co/spaces/felixrosberg/face-swap). The Space allow you to face swap images, but also try some other functionality I am currently researching, which I plan to publish soon. For example, reconstruction attacks and adversarial defense against the reconstruction attacks.



## Getting Started
This project was implemented in TensorFlow 2.X. For evaluation we used models implemented in both TensorFlow and PyTorch (e.g CosFace from [InsightFace](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch)).

### Installation:
- Install depencies:


```shell
pip install -r requirements.txt
```


#### An alternative installation method if you have difficulty with the previous:
- Clone or download repository
```shell
git clone https://github.com/felixrosberg/FaceDancer.git
cd FaceDancer
```

- Make conda environment
```shell
conda create -n facedancer python=3.8
conda activate facedancer
python -m pip install --upgrade pip
```

- Install depencies:
```shell
conda install -c conda-forge cudatoolkit cudnn
conda install -c conda-forge ffmpeg

pip install tensorflow-gpu
pip install -r requirements.txt
```

