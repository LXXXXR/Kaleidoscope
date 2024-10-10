
# Kaleidoscope: Learnable Masks for Heterogeneous Multi-agent Reinforcement Learning

This is the implementation of our paper "[Kaleidoscope: Learnable Masks for Heterogeneous Multi-agent Reinforcement Learning]()" in NeurIPS 2024. This repo is based on benellis3's [fork](https://github.com/benellis3/pymarl2) of the open-source [pymarl2](https://github.com/hijkzzz/pymarl2). Please refer to those repo for more documentation.

## Installation instructions

**Create Conda Environment**

Install Python environment with conda:
```bash
conda create -n kaleidoscope python=3.8
conda activate kaleidoscope
```

**Install Dependencies and Setup SMAC**
```bash
cd src
bash install_dependecies.sh
bash install_sc2.sh
```

To ease the environment setup, we also provide the environmental setup we used containing detailed version information in `Kalei_SMACv2_Env.txt`. 


## Run an experiment 

```shell
python src/main.py --config=[Algorithm name] --env-config=[Env name] --exp-config=[Experiment name]
```

The config files are all located in `src/config`.

`--config` refers to the config files in `src/config/algs`.
`--env-config` refers to the config files in `src/config/envs`.
`--exp-config` refers to the config files in `src/config/exp`. If you want to change the configuration of a particular experiment, you can do so by modifying the yaml file here.

All results will be stored in the `work_dirs` folder.

For example, run Kaleidoscope on Zerg5v5:

```shell
python src/main.py --config=Kalei_qmix_rnn_1R3 --env-config=sc2_gen_zerg --exp-config=zerg_5v5_10M_s0
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