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