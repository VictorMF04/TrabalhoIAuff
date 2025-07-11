#!/usr/bin/env python3
"""
Script de inferência GraphWalks usando GPT-3.5-turbo via OpenAI API (openai>=1.0.0).
Requer chave da API configurada como variável de ambiente OPENAI_API_KEY.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# Carrega variáveis do .env, se existir
load_dotenv()

# Inicializar cliente OpenAI
client = OpenAI()

# Prompt completo
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

print("[DEBUG] Enviando prompt para GPT-3.5...")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ],
    temperature=0.7,
    max_tokens=128,
    top_p=0.9,
    n=1
)

generated = response.choices[0].message.content.strip()
print("[DEBUG] Resposta gerada:\n", generated)

# Extrair resposta
if "Final Answer:" in generated:
    answer = generated.split("Final Answer:")[-1].strip()
else:
    answer = generated.strip()
