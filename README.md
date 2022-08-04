# Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation

**Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation** <br />
[Shengyu Zhang](https://shengyuzhang.github.io/), Lingxiao Yang, Dong Yao and Yujie Lu, [Fuli Feng](https://fulifeng.github.io/), [Zhou Zhao](https://person.zju.edu.cn/zhaozhou), [Tat-Seng Chua](https://www.chuatatseng.com/), [Fei Wu](https://person.zju.edu.cn/en/wufei) <br />
**The ACM Web Conference 2022 (WWW 2022)** <br />
**Key Words: &nbsp;Recommender Systems; &nbsp;Multi-interest; &nbsp;Backward Flow** <br />
**[[Paper](https://dl.acm.org/doi/10.1145/3485447.3512094)]**, **[[Slides](https://zjueducn-my.sharepoint.com/:b:/g/personal/sy_zhang_zju_edu_cn/EZ4YnJfnWJNPkGAlpq_75OkB1XzygJyJWh7RwQ7f16u_8Q?e=66ujzY)]** <br />


## A pytorch implementation of Re4


## Prerequisites

- Python 3
- PyTorch 1.8.1
- TensorFlow 2.x

## Getting Started

### Installation

- Install PyTorch 1.8.1
- Install TensorFlow 2.x
- Clone this repository `git clone https://github.com/DeerSheep0314/Re4-Learning-to-Re-contrast-Re-attend-Re-construct-for-Multi-interest-Recommendation.git`.

### Dataset

- Amazon-book dataset can be downloaded through:
  - Microsoft OneDrive [[Link](https://zjueducn-my.sharepoint.com/:f:/g/personal/sy_zhang_zju_edu_cn/EpU49hRo6jVPn0o2x5tvM90BZO8KCo9UqHE8N2bZZnGuMA?e=8UBAgt)]

### Running

To run the code, You can use `python src/model.py --gpu {gpu_num} --thre {thre_num} --data {dataset_name} --ct_lambda {ct_weight} --cs_lambda {cs_weight} --att_lambda {att_weight} --numin {num_interests}` to train the R4 model on a specific dataset. You can set the above hyperparameters here, see the code for other hyperparameters.

For example, you can use `python src/model.py --gpu 0 --thre -1 --numin 8 --data book --ct_lambda 0.1 --cs_lambda 0.1 --att_lambda 0.001` to train R4 model on Amazon-book dataset.

## Bibtex
```
@inproceedings{DBLP:conf/www/ZhangYYLFZC022,
  author    = {Shengyu Zhang and
               Lingxiao Yang and
               Dong Yao and
               Yujie Lu and
               Fuli Feng and
               Zhou Zhao and
               Tat{-}Seng Chua and
               Fei Wu},
  title     = {Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest
               Recommendation},
  booktitle = {{WWW} '22: The {ACM} Web Conference 2022},
  pages     = {2216--2226},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3485447.3512094},
}
```
