# PEOA-CTA

Official implementation of **Beyond Visual Prompt: Unlocking Parameter-Efficient Online Adapter for Continual Test-Time Adaptation in Medical Image Segmentation**.

PEOA-CTA adapts a frozen segmentation model to a sequence of target domains through lightweight online adapters. The method combines multi-scale feature extraction, memory retrieval, dual-reliability pseudo-label selection, and batch-normalization statistics alignment.

![PEOA-CTA overview](assets/intro.png)

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate peoa-cta
```

The configured environment uses PyTorch 2.3.0 and CUDA 12.1.

## Dataset

Download and extract the fundus dataset:

```bash
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/data_lx/Fundus.zip
unzip Fundus.zip -d data
```

The dataset root must contain the dataset CSV files. Each CSV must provide `image` and `mask` columns with paths relative to the dataset root.

Supported domain names are:

- `RIM_ONE_r3`
- `REFUGE`
- `ORIGA`
- `REFUGE_Valid`
- `Drishti_GS`

## Source model

Model weights are not included in this repository. Place a source checkpoint at:

```text
models/<SOURCE_DATASET>/last-Res_Unet.pth
```

A source model can also be trained with:

```bash
cd OPTIC
python train_source.py \
  --Source_Dataset ORIGA \
  --dataset_root ../data/Fundus \
  --path_save_model ../models
```

## Run

The default command uses ORIGA as the source domain and RIM-ONE-r3 as the target domain:

```bash
bash PEOA_cta_optic.sh
```

Paths and domains can be configured through environment variables:

```bash
DATASET_ROOT=/path/to/Fundus \
MODEL_ROOT=/path/to/models \
LOG_ROOT=/path/to/logs \
SOURCE_DATASET=ORIGA \
TARGET_DATASETS=RIM_ONE_r3 \
bash PEOA_cta_optic.sh
```

Multiple target domains can be passed in stream order:

```bash
cd OPTIC
python BBA.py \
  --dataset_root ../data/Fundus \
  --model_root ../models \
  --Source_Dataset ORIGA \
  --target_datasets RIM_ONE_r3 REFUGE Drishti_GS
```

## Citation

```bibtex
@article{li2026peoa,
  title   = {Beyond Visual Prompt: Unlocking Parameter-Efficient Online Adapter for Continual Test-Time Adaptation in Medical Image Segmentation},
  author  = {Li, Xiang and Fang, Huihui and Liu, Mingsi and Wang, Jinghao and Duan, Lixin and Fang, Yuqi and Xu, Yanwu},
  journal = {Information Fusion},
  volume  = {133},
  pages   = {104256},
  year    = {2026},
  doi     = {10.1016/j.inffus.2026.104256}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

This repository builds on [VPTTA](https://github.com/Chen-Ziyang/VPTTA), [DLTTA](https://github.com/med-air/DLTTA), and [DomainAdaptor](https://github.com/koncle/DomainAdaptor).
