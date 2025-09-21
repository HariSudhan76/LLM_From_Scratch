#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import tensorflow as tf
import tqdm
import torch
import nbimporter
from GPT_Architecture import GPTModel
import sys
import os
notebook_dir = os.getcwd()
llm_eval_path = os.path.abspath(os.path.join(notebook_dir, "..", "4.LLM_Evaluation"))

sys.path.append(llm_eval_path)
import LLMPerformance as llmp

print(f"tensorflow version: {tf.__version__}")
print(f"tqdm version: {tqdm.__version__}")


# In[22]:


GPT_CONFIG_124M={
 "vocab_size":50257,
 "context_length":1024,
 "emb_dim":768,
 "n_heads":12,
 "n_layers":12,
 "drop_rate":0.1,
 "qkv_bias":False }


# In[23]:


from gpt_download3 import download_and_load_gpt2


# In[24]:


settings, params = download_and_load_gpt2(model_size="124M",models_dir="gpt2")


# In[25]:


print("Settings:",settings)
print("Parameter dictionary keys:",params.keys())
##Both settings are params - Python dictionaries. The settings stores the LLM architecure settings similarly to 
## GPT_CONFIG_124M settings

# The params dictionary contains the actual weight tensors
# Only printed the dic keys because print the weight contents would take up too much screen space


# In[26]:


print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)


# In[27]:


#Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers":12, "n_heads": 12},
    "gpt2-medium (335M)":{"emb_dim":1024, "n_layers":24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers":36, "n_heads": 20},
    "gpt2-xl (1558M)":{"emb_dim":1600, "n_layers":48, "n_heads": 25},
}

#Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])


# In[28]:


NEW_CONFIG.update({"context_length": 1024, "qkv_bias":True}) ##Bias vecotrs not going to perform any better for 
# consistancy with gpt2 model we are enabling the qkv_bias and setting it to True
gpt = GPTModel(NEW_CONFIG)
gpt.eval()


# In[29]:


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


# In[30]:


def load_weights_into_gpt(gpt, params):
    # Embeddings
    assign(gpt.pos_emb.weight, params['wpe'])
    assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        block = params["blocks"][b]

        # ---- Attention qkv ----
        q_w, k_w, v_w = np.split(block["attn"]["c_attn"]["w"], 3, axis=-1)
        assign(gpt.trf_blocks[b].attn.W_query.weight, q_w.T)
        assign(gpt.trf_blocks[b].attn.W_key.weight,   k_w.T)
        assign(gpt.trf_blocks[b].attn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(block["attn"]["c_attn"]["b"], 3, axis=-1)
        assign(gpt.trf_blocks[b].attn.W_query.bias, q_b)
        assign(gpt.trf_blocks[b].attn.W_key.bias,   k_b)
        assign(gpt.trf_blocks[b].attn.W_value.bias, v_b)

        # ---- Attention output projection ----
        assign(gpt.trf_blocks[b].attn.out_proj.weight, block["attn"]["c_proj"]["w"].T)
        assign(gpt.trf_blocks[b].attn.out_proj.bias,   block["attn"]["c_proj"]["b"])

        # ---- Feed-forward ----
        assign(gpt.trf_blocks[b].ff.layers[0].weight, block["mlp"]["c_fc"]["w"].T)
        assign(gpt.trf_blocks[b].ff.layers[0].bias,   block["mlp"]["c_fc"]["b"])
        assign(gpt.trf_blocks[b].ff.layers[2].weight, block["mlp"]["c_proj"]["w"].T)
        assign(gpt.trf_blocks[b].ff.layers[2].bias,   block["mlp"]["c_proj"]["b"])

        # ---- LayerNorms ----
        assign(gpt.trf_blocks[b].norm1.scale, block["ln_1"]["g"])
        assign(gpt.trf_blocks[b].norm1.shift, block["ln_1"]["b"])
        assign(gpt.trf_blocks[b].norm2.scale, block["ln_2"]["g"])
        assign(gpt.trf_blocks[b].norm2.shift, block["ln_2"]["b"])

    # Final norm + head
    assign(gpt.final_norm.scale, params["g"])
    assign(gpt.final_norm.shift, params["b"])

    # Tie output head to token embeddings
    assign(gpt.out_head.weight, params["wte"])


# In[31]:


import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
load_weights_into_gpt(gpt,params)
gpt.to(device)


# In[32]:


load_weights_into_gpt(gpt,params)
gpt.to(device)


# In[38]:


torch.manual_seed(123)

token_ids = llmp.generate(
    model=gpt,
    idx=llmp.text_to_token_ids("fear of Napoleon's", llmp.tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=10
)
print("Output text:\n",llmp.token_ids_to_text(token_ids, llmp.tokenizer))


# In[34]:


import sys
get_ipython().system('"{sys.executable}" -m pip install pypdf')


