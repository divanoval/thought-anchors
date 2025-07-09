# interactive_demo.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import split_solution_into_chunks, get_chunk_ranges
import circuitsvis as cv
from whitebox_analyses.model_read import ActivationCollector
import numpy as np
# Should I eat eggs or beef if I care about the environment?

# # Python Example
# from circuitsvis.tokens import colored_tokens
# colored_tokens(["My", "tokens"], [0.123, -0.226])

# 1) Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True).to("cuda")

# 2) Get a prompt from the user and generate a chain of thought
prompt = "Should I eat eggs or beef if I care about the environment?"#input("Prompt: ")
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

# Create attention mask (1 for real tokens, 0 for padding)
attention_mask = torch.ones_like(input_ids)

# First generate without attention outputs
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,  # Use EOS as PAD since DeepSeek doesn't have PAD
        return_dict_in_generate=True
    ).sequences

# Now do a forward pass to get attention patterns
full_attention_mask = torch.ones((1, generated_ids.shape[1]), device=model.device)

with torch.no_grad():
    outputs = model(
        generated_ids,
        attention_mask=full_attention_mask,
        output_attentions=True,
        return_dict=True
    )
    attn_weights = outputs.attentions

text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
sentences = split_solution_into_chunks(text)
print("Generated sentences:", sentences)

# 3) Get token ranges for each sentence
chunk_ranges = get_chunk_ranges(text, sentences)
print("Token ranges for each sentence:", chunk_ranges)

# 4) Compute per-head sentence-sentence matrices
num_layers = len(attn_weights)
num_heads = attn_weights[0].shape[1]  # Each layer's attention has shape (batch, heads, seq, seq)
print(f"Number of layers: {num_layers}, Number of heads per layer: {num_heads}")
num_sentences = len(sentences)

# Initialize sentence-level attention tensor
sentence_attn = torch.zeros(num_layers, num_heads, num_sentences, num_sentences)

# Aggregate token-level attention into sentence-level attention
for layer in range(num_layers):
    layer_attn = attn_weights[layer][0]  # Remove batch dimension
    for head in range(num_heads):
        head_attn = layer_attn[head]
        
        # For each pair of sentences, average the attention weights between their tokens
        for i, (start_i, end_i) in enumerate(chunk_ranges):
            for j, (start_j, end_j) in enumerate(chunk_ranges):
                # Extract and average attention weights for this sentence pair
                sentence_pair_attn = head_attn[start_i:end_i, start_j:end_j]
                sentence_attn[layer, head, i, j] = sentence_pair_attn.mean()

# 5) For each layer and head, show a heatmap with circuitsvis
for layer in range(num_layers):
    for head in range(num_heads):
        print(f"\nLayer {layer}, Head {head} Attention Pattern:")
        heatmap = cv.attention.attention_patterns(
            sentence_attn[layer, head].cpu().numpy(),
            x_axis=sentences,
            y_axis=sentences,
            title=f"Layer {layer}, Head {head}"
        )
        heatmap.show()

# 5) Optionally compute causal importance matrix using compute_step_importance_matrix
