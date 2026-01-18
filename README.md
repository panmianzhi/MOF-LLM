# Enhancing Spatial Reasoning in Large Language Models for Metal-Organic Frameworks Structure Prediction
[![arXiv](https://img.shields.io/badge/arXiv-2601.09285-b31b1b.svg)](https://arxiv.org/abs/2601.09285)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-MOF--LLM-yellow.svg)](https://huggingface.co/MianzhiPan/MOF-LLM)

## Data
The MOF structure dataset is from [MOFFlow](https://github.com/nayoung10/MOFFlow). Our processed pre-training, sft and rl prompts can be download from [Google Drive](https://drive.google.com/drive/folders/1pmQxoHW6gQfAzSwrZmOULOfvv8tSpK4a?usp=drive_link).

## Inference code & Model checkpoint
The MOF-LLM's checkpoint can be download from [huggingface](https://huggingface.co/MianzhiPan/MOF-LLM).

## Evaluation
Code for calculating the rmsd and match rate is in `evaluate.py`.

## Citation
```
@misc{pan2026enhancingspatialreasoninglarge,
      title={Enhancing Spatial Reasoning in Large Language Models for Metal-Organic Frameworks Structure Prediction}, 
      author={Mianzhi Pan and JianFei Li and Peishuo Liu and Botian Wang and Yawen Ouyang and Yiming Rong and Hao Zhou and Jianbing Zhang},
      year={2026},
      eprint={2601.09285},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.09285}, 
}
```