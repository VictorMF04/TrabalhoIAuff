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

# Prompt completo com exemplo 1 do dataset
prompt_text = """
Here is the graph to operate on:
The graph has the following edges:
cfcd208495 -> cfcd208495
cfcd208495 -> 1679091c5a
cfcd208495 -> c81e728d9d
cfcd208495 -> c4ca4238a0
c4ca4238a0 -> c9f0f895fb
c4ca4238a0 -> 45c48cce2e
c4ca4238a0 -> eccbc87e4b
c4ca4238a0 -> c9f0f895fb
c81e728d9d -> 45c48cce2e
c81e728d9d -> eccbc87e4b
c81e728d9d -> eccbc87e4b
c81e728d9d -> c9f0f895fb
eccbc87e4b -> d3d9446802
eccbc87e4b -> d3d9446802
eccbc87e4b -> a87ff679a2
eccbc87e4b -> c4ca4238a0
a87ff679a2 -> 1679091c5a
a87ff679a2 -> eccbc87e4b
a87ff679a2 -> cfcd208495
a87ff679a2 -> e4da3b7fbb
e4da3b7fbb -> c4ca4238a0
e4da3b7fbb -> 1679091c5a
e4da3b7fbb -> 1679091c5a
e4da3b7fbb -> 45c48cce2e
1679091c5a -> 8f14e45fce
1679091c5a -> 8f14e45fce
1679091c5a -> e4da3b7fbb
1679091c5a -> a87ff679a2
8f14e45fce -> 45c48cce2e
8f14e45fce -> e4da3b7fbb
8f14e45fce -> 8f14e45fce
8f14e45fce -> cfcd208495
c9f0f895fb -> eccbc87e4b
c9f0f895fb -> cfcd208495
c9f0f895fb -> eccbc87e4b
c9f0f895fb -> 8f14e45fce
45c48cce2e -> cfcd208495
45c48cce2e -> 1679091c5a
45c48cce2e -> a87ff679a2
45c48cce2e -> a87ff679a2
d3d9446802 -> 8f14e45fce
d3d9446802 -> d3d9446802
d3d9446802 -> 45c48cce2e
d3d9446802 -> e4da3b7fbb

Operation:
Find the parents of node 8f14e45fce.

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

