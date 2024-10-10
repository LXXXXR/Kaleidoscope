# Kaleidoscope: Learnable Masks for Heterogeneous Multi-agent Reinforcement Learning

This is the implementation of our paper "[Kaleidoscope: Learnable Masks for Heterogeneous Multi-agent Reinforcement Learning]()" in NeurIPS 2024. This repo is based on the open-source [HARL](https://github.com/PKU-MARL/HARL). Please refer to that repo for more documentation.

## Installation instructions

**Install MuJoCo**

First, follow the instructions on https://github.com/openai/mujoco-py, https://www.roboti.us/, and https://github.com/deepmind/mujoco to download the right version of mujoco you need.

Second, `mkdir ~/.mujoco`.

Third, move the .tar.gz or .zip to `~/.mujoco`, and extract it using `tar -zxvf` or `unzip`.

Fourth, add the following line to the `.bashrc`:

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<user>/.mujoco/<folder-name, e.g. mujoco210, mujoco-2.2.1>/bin
```

Fifth, run the following command:

```shell
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
pip install mujoco
pip install gym[mujoco]
sudo apt-get update -y
sudo apt-get install -y patchelf
```

**Install Dependencies of MAMuJoCo**

First follow the instructions above to install MuJoCo. Then run the following commands.

```shell
pip install "mujoco-py>=2.1.2.14"
pip install "Jinja2>=3.0.3"
pip install "glfw>=2.5.1"
pip install "Cython>=0.29.28"
```

Note that [mujoco-py](https://github.com/openai/mujoco-py) is compatible with `mujoco210` (see [this](https://github.com/openai/mujoco-py#install-mujoco)). So please make sure to download `mujoco210` and extract it into the right place.


**Install HARL**

Install Python environment with conda:

```bash
conda create -n kaleidoscope python=3.8
conda activate kaleidoscope
pip install -e .
```

To ease the environment setup, we also provide the environmental setup we used containing detailed version information in `Kalei_MaMuJoCo_Env.txt`. 

## Run an experiment 

```shell
cd src
python examples/train.py --load_config=[Experiment name]
```


All results will be stored in the `work_dirs` folder.

For example, run Kaleidoscope on Ant-v2-4x2:

```shell
python examples/train.py --load_config tuned_configs/Ant-v2-4x2-Kalei_matd3_s0.json
```


## Citing

If you use this code in your research or find it helpful, please consider citing our paper:
```
@article{li2024kaleidoscope,
  title={Kaleidoscope: Learnable Masks for Heterogeneous
Multi-agent Reinforcement Learning},
  author={Li, Xinran and Pan, Ling and Zhang, Jun},
  booktitle={accepted by the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```