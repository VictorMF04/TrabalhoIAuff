from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os

# 1. Configurações gerais
MODEL_NAME  = "microsoft/phi-1_5"
OUT_DIR     = "./phi_graphwalks_lora"
MAX_LEN     = 512      
FRAC        = 0.05     # 5% do dataset
EPOCHS      = 3
BATCH_SIZE  = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Carrega e formata o dataset
ds = load_dataset("openai/graphwalks", split="train")
ds = ds.shuffle(seed=42).select(range(int(len(ds) * FRAC)))

def format_example(ex):
    ans = "[" + ", ".join(ex["answer_nodes"]) + "]"
    return {"text": f"{ex['prompt'].strip()}\nResposta: {ans}"}

ds = ds.map(format_example).remove_columns([c for c in ds.column_names if c != "text"])
split = ds.train_test_split(test_size=0.1)
train_ds, val_ds = split["train"], split["test"]

# 3. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
val_ds   = val_ds.map(tokenize_fn,  batched=True, remove_columns=["text"])

# 4. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Carrega o modelo sem quantização 4‑bit
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if device == "cuda" else None
)
model = prepare_model_for_kbit_training(model)

# 6. Configura LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 7. Treinamento
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    load_best_model_at_end=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator
)
trainer.train()
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"Adaptadores LoRA salvos em {OUT_DIR}")

# 8. Inferência com Prompt Customizado e DEBUG
prompt_text = """
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
Perform a BFS from node alke with depth 1.
Final Answer: []

Example 3:
Operation:
Find the parents of node uvwx.
Final Answer: [abcd, efgh]

Example 4:
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

You should immediately return the set of nodes that the operation results in, with no additional text. Return your final answer as a list of nodes in the very last line of your response.
"""

# Tokeniza e move para GPU
inputs = tokenizer(
    prompt_text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LEN
).to(model.device)

print(">> Prompt enviado ao modelo:")
print(prompt_text)

# Garante modo avaliação e limpa cache da GPU
model.eval()
torch.cuda.empty_cache()

# Gera pelo menos 1 token novo, sem parar no eos interno
max_length = inputs["input_ids"].shape[-1] + 64
with torch.no_grad():
    gen_ids = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        min_new_tokens=1,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        num_return_sequences=1,
    )

decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
print(">> Textos gerados completos:")
print(decoded)

if "Final Answer:" in decoded:
    answer = decoded.split("Final Answer:")[-1].strip()
else:
    answer = decoded[len(prompt_text):].strip()

print("Resposta do modelo:", answer)
