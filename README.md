# Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation

<!-- **Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation** <br />
[Shengyu Zhang](https://shengyuzhang.github.io/), Lingxiao Yang, Dong Yao and Yujie Lu, [Fuli Feng](https://fulifeng.github.io/), [Zhou Zhao](https://person.zju.edu.cn/zhaozhou), [Tat-Seng Chua](https://www.chuatatseng.com/), [Fei Wu](https://person.zju.edu.cn/en/wufei) <br />
**The ACM Web Conference 2022 (WWW 2022)** <br />
**Key Words: &nbsp;Recommender Systems; &nbsp;Multi-interest; &nbsp;Backward Flow** <br />
**[[Paper](https://dl.acm.org/doi/10.1145/3485447.3512094)]**, **[[Slides](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/84d2c2ba-4a10-4b75-9788-3069a453c671/WWW_Re4.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220518%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220518T011944Z&X-Amz-Expires=86400&X-Amz-Signature=17f033dd56e705f10b9248ff8a0851d05d159e810d0cd204f5e10a1c61b7fc1c&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22WWW_Re4.pdf%22&x-id=GetObject)]** <br /> -->


## A pytorch implementation of Re4

Will pick up the code refracting procedure if we have a time table. Please kindly wait for the release.

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
  - Microsoft OneDrive link : 

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
