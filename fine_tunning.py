from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

# Configurações
MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./output_model"
MAX_LENGTH = 128

# Carrega 5% do dataset para teste inicial (pode aumentar depois)
dataset = load_dataset("openai/graphwalks", split="train")

# Formata para o modelo aprender: prompt + answer
def format_example(example):
    # answer_nodes é uma lista, vamos transformar em string tipo "[node1, node2]"
    answer_str = "[" + ", ".join(example["answer_nodes"]) + "]"
    return {
        "text": f"{example['prompt'].strip()}\nResposta: {answer_str}"
    }

dataset = dataset.map(format_example)
dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

# Divide em treino e validação (90% treino, 10% val)
split = dataset.train_test_split(test_size=0.1)
train_dataset = split["train"]
eval_dataset = split["test"]

# Carrega tokenizer e ajusta pad_token se necessário
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

# Tokeniza os datasets
train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

# Remove coluna texto para treinamento (mantém só os tensores)
train_dataset = train_dataset.remove_columns(["text"])
eval_dataset = eval_dataset.remove_columns(["text"])

# Data collator para causal LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Carrega o modelo
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Configurações de treino
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    do_eval=True,        # executa avaliação
    do_train=True,       # executa treino (é o default)
    eval_steps=500,      # passo em que a cada 500 batches roda validação
    save_steps=500,      # passo em que salva checkpoint
    save_total_limit=2, 
    eval_strategy="epoch",  # Avalia a cada epoch
    save_strategy="epoch",        # Salva checkpoint a cada epoch
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    weight_decay=0.01,          # Mantém só os 2 últimos checkpoints
    load_best_model_at_end=True,  # Carrega o melhor modelo ao final
    metric_for_best_model="loss",
    greater_is_better=False,  # Métrica para escolher melhor modelo
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Treinamento
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Modelo fine-tuned salvo em: {OUTPUT_DIR}")
