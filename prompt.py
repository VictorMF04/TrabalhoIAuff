from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Caminho do modelo fine-tuned
MODEL_PATH = "./output_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Prompt de exemplo se o usuário não digitar nada
default_prompt = """
You will be given a graph as a list of directed edges. All nodes are at least degree 1. You will also get a description of an operation to perform on the graph. 
Your job is to execute the operation on the graph and return the set of nodes that the operation results in. If asked for a breadth-first search (BFS), only return the nodes that are reachable at that depth, do not return the starting node. If asked for the parents of a node, only return the nodes that have an edge leading to the given node, do not return the given node itself.

Here is an example:
<begin example>

The graph has the following edges:
uvwx -> alke
abcd -> uvwx
abcd -> efgh
efgh -> uvwx

Example 1:
Operation:
Perform a BFS from node abcd with depth 1.
Final Answer: [uvwx, efgh]

Example 2:
Operation:
Find the parents of node abcd.
Final Answer: []

<end example>
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

You should immediately return the set of nodes that the operation results in, with no additional text. Return your final answer as a list of nodes in the very last line of your response. For example, if the operation returns the set of nodes [node1, node2, node3], your response should be:
Final Answer: [node1, node2, node3]
If the operation returns the empty set, your response should be:
Final Answer: []
""".strip()

# Entrada via terminal
print("Digite um prompt ou pressione Enter para usar o exemplo:")
user_input = input("> ").strip()
prompt = user_input if user_input else default_prompt

# Tokenização
tokens = tokenizer(prompt,
                   truncation=True,
                   max_length=1024,
                   return_tensors="pt",
                   padding=False,
                   return_attention_mask=True)

inputs = tokenizer(prompt, return_tensors="pt")
model.config.pad_token_id = tokenizer.eos_token_id
generate_kwargs = dict(
    **tokens,                   # input_ids e attention_mask já como tensores
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
)
with torch.no_grad():
    outputs = model.generate(**generate_kwargs)

# 3) Chamada ao modelo
outputs = model.generate(**generate_kwargs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Decodificação
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n📤 Resposta gerada:")
print(generated[len(prompt):].strip())
