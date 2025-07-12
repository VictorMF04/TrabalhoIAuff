#!/usr/bin/env python3
"""
Script de inferência GraphWalks usando GPT-2.
Carrega modelo GPT-2 padrão e executa prompt completo.
"""
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("hello")
print(f"[DEBUG] Python: {sys.executable}")

# Configurações
MODEL_NAME = "gpt2"        # usar GPT-2 padrão
MAX_CONTEXT = 1024          # limite de contexto do modelo
MAX_NEW_TOKENS = 64         # quantidade de tokens para gerar
MAX_INPUT_LEN = MAX_CONTEXT - MAX_NEW_TOKENS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] Dispositivo: {device}")

# Carrega modelo GPT-2
print(f"[DEBUG] Carregando modelo {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
print("[DEBUG] Modelo carregado.")

# Carrega tokenizador
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# GPT-2 não possui token de pad por padrão
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("[DEBUG] Tokenizador carregado.")

# Prompt completo com exemplo 6 do dataset
prompt_text = """
Here is the graph to operate on:
The graph has the following edges:
cfcd208495 -> 45c48cce2e
c4ca4238a0 -> c74d97b01e
c81e728d9d -> 1c383cd30b
eccbc87e4b -> d645920e39
a87ff679a2 -> e369853df7
e4da3b7fbb -> c74d97b01e
1679091c5a -> 70efdf2ec9
8f14e45fce -> d645920e39
c9f0f895fb -> aab3238922
45c48cce2e -> 1f0e3dad99
d3d9446802 -> d645920e39
6512bd43d9 -> 34173cb38f
c20ad4d76f -> eccbc87e4b
c51ce410c1 -> 182be0c5cd
aab3238922 -> a1d0c6e83f
9bf31c7ff0 -> a5771bce93
c74d97b01e -> 70efdf2ec9
70efdf2ec9 -> aab3238922
6f4922f455 -> c4ca4238a0
1f0e3dad99 -> f7177163c8
98f1370821 -> c20ad4d76f
3c59dc048e -> aab3238922
b6d767d2f8 -> 4e732ced34
37693cfc74 -> e369853df7
1ff1de7740 -> 33e75ff09d
8e296a067a -> 3c59dc048e
4e732ced34 -> c81e728d9d
02e74f10e0 -> 6ea9ab1baa
33e75ff09d -> c74d97b01e
6ea9ab1baa -> 98f1370821
34173cb38f -> 37693cfc74
c16a5320fa -> 33e75ff09d
6364d3f0f4 -> 3c59dc048e
182be0c5cd -> a87ff679a2
e369853df7 -> 8e296a067a
1c383cd30b -> a5771bce93
19ca14e7ea -> d3d9446802
a5bfc9e079 -> 1c383cd30b
a5771bce93 -> 98f1370821
d67d8ab4f4 -> d3d9446802
d645920e39 -> 9bf31c7ff0
3416a75f4c -> c9f0f895fb
a1d0c6e83f -> 182be0c5cd
17e62166fc -> 182be0c5cd
f7177163c8 -> c16a5320fa
6c8349cc72 -> 17e62166fc

Operation:
Find the parents of node 3416a75f4c.

You should immediately return the set of nodes that the operation results in, with no additional text. Return your final answer as a list of nodes in the very last line of your response.
"""
print(f"[DEBUG] Prompt length: {len(prompt_text)} chars")

# Tokenização com truncamento para garantir espaço para geração
tokens = tokenizer(
    prompt_text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_INPUT_LEN
)
inputs = tokens.to(device)
print(f"[DEBUG] Input IDs shape: {inputs['input_ids'].shape}")

# Calcular espaço seguro para geração
max_new = min(MAX_NEW_TOKENS, MAX_CONTEXT - inputs['input_ids'].shape[-1])

# Inferência
gen_ids = model.generate(
    **inputs,
    max_new_tokens=max_new,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    num_return_sequences=1
)
print(f"[DEBUG] Gen IDs shape: {gen_ids.shape}")

# Decodificar e exibir
decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
print("[DEBUG] Generated text:\n", decoded)

# Extrair resposta
if "Final Answer:" in decoded:
    answer = decoded.split("Final Answer:")[-1].strip()
else:
    answer = decoded[len(prompt_text):].strip()

