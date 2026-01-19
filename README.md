# Enhancing Spatial Reasoning in Large Language Models for Metal-Organic Frameworks Structure Prediction
[![arXiv](https://img.shields.io/badge/arXiv-2601.09285-b31b1b.svg)](https://arxiv.org/abs/2601.09285)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-MOF--LLM-yellow.svg)](https://huggingface.co/MianzhiPan/MOF-LLM)

## Data
The MOF structure dataset is from [MOFFlow](https://github.com/nayoung10/MOFFlow). Our processed pre-training, sft and rl prompts can be download from [Google Drive](https://drive.google.com/drive/folders/1pmQxoHW6gQfAzSwrZmOULOfvv8tSpK4a?usp=drive_link).

## Inference code & Model checkpoint
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "MianzhiPan/MOF-LLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

messages = [
    {"role": "system", "content": "You are tasked with predicting the 3D crystal structure of a Metal-Organic Framework (MOF). MOFs are highly ordered porous materials formed by connecting metal-containing nodes with organic linkers, creating modular structures. \nGiven the SMILES of all MOF building blocks, predict the complete 3D crystal structure configuration including:\nLattice parameters\u200b in the format 'a b c \u03b1 \u03b2 \u03b3' (where a, b, c are unit cell lengths, and \u03b1, \u03b2, \u03b3 are unit cell angles). \nFor each building block\u200b (maintaining the exact same order as provided in the input):\nTranslation vector\u200b ([tx ty tz]): The position of the building block's center within the unit cell, expressed in fractional coordinates.\nRotation angles\u200b ([roll pitch yaw]): The orientation of the building block, represented in radians\u200b using Euler angles.\nOutput Format:\nFirst line: Lattice parameters, i.e.: a b c \u03b1 \u03b2 \u03b3.\nSubsequent lines (one per building block): [k] tx ty tz roll pitch yaw\nk: The 0-based index of the building block.\ntx ty tz: Fractional coordinates within the unit cell.\nroll pitch yaw: Euler angles in radians.\n"},
    {"role": "user", "content": "Input Building Blocks(Separate by spaces): [Zn-][Zn-] O=C([O-])c1ccc(-c2ccc(C(=O)[O-])cc2)cc1 O=C([O-])c1ccc(C#Cc2ccc(C(=O)[O-])cc2)cc1 CCCOc1cncc(OCCC)c1-c1c(OCCC)cncc1OCCC"}
]

formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.1, top_p=0.6, top_k=10)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

We recommend to use vllm for large-scale inference. Below is an inference script on our demo data
```bash
cd scripts
bash vllm_infer.sh
```
To recover the all-atom MOF structure, please refer to `evaluate.prompt_to_structure` method.

## Evaluation
Code for calculating the rmsd and match rate is in `evaluate.py`.

## Citation
```bibtex
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