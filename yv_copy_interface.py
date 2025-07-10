# interactive_demo.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import circuitsvis as cv
import re
from utils import split_solution_into_chunks, get_chunk_ranges
from whitebox_analyses.model_read import ActivationCollector
import numpy as np
from scipy import stats

# Configuration constants
MIN_SENTENCE_DISTANCE = 4  # Minimum distance between sentences for attention analysis

def avg_matrix_by_chunk(matrix, chunk_token_ranges):
    n = len(chunk_token_ranges)
    avg_mat = np.zeros((n, n), dtype=np.float32)
    for i, (start_i, end_i) in enumerate(chunk_token_ranges):
        for j, (start_j, end_j) in enumerate(chunk_token_ranges):
            region = matrix[start_i:end_i, start_j:end_j]
            if region.size > 0:
                avg_mat[i, j] = region.mean().item()
    return avg_mat

def get_attn_vert_scores(avg_mat, proximity_ignore=4, drop_first=0):
    n = avg_mat.shape[0]
    vert_scores = []
    for i in range(n):
        vert_lines = avg_mat[i + proximity_ignore:, i]
        vert_score = np.nanmean(vert_lines) if len(vert_lines) > 0 else np.nan
        vert_scores.append(vert_score)
    vert_scores = np.array(vert_scores)
    if drop_first > 0:
        vert_scores[:drop_first] = np.nan
        vert_scores[-drop_first:] = np.nan
    return vert_scores

def get_model_and_tokenizer(model=None, tokenizer=None):
    """Helper function to get or reuse model and tokenizer instances.
    
    Args:
        model: Optional pre-loaded model instance
        tokenizer: Optional pre-loaded tokenizer instance
    
    Returns:
        tuple: (model, tokenizer) instances
    """
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    
    if model is None or tokenizer is None:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True).to("cuda")
            print("Loaded new model instance")
    else:
        print("Using pre-loaded model instance")
    
    return model, tokenizer

def generate_text_and_get_attention(model, tokenizer, prompt):
    """Generate text and get attention patterns for the input prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text prompt
    
    Returns:
        tuple: (generated_ids, attention_weights, decoded_text)
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)

    # First generate without attention outputs
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        ).sequences

    # Get attention patterns
    full_attention_mask = torch.ones((1, generated_ids.shape[1]), device=model.device)
    with torch.no_grad():
        outputs = model(
            generated_ids,
            attention_mask=full_attention_mask,
            output_attentions=True,
            return_dict=True
        )
    
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_ids, outputs.attentions, text

def analyze_sentence_pair_attention(head_attn, start_i, end_i, start_j, end_j, head, i, j):
    """Analyze attention between a pair of sentences.
    
    Args:
        head_attn: Attention weights for current head
        start_i, end_i: Token range for source sentence
        start_j, end_j: Token range for target sentence
        head, i, j: Indices for head and sentences
    
    Returns:
        dict: Information about the attention pair
    """
    # Check for empty ranges
    if start_i >= end_i or start_j >= end_j:
        return {
            'head': head,
            'sentence_i': i,
            'sentence_j': j,
            'range_i': (start_i, end_i),
            'range_j': (start_j, end_j),
            'attention_shape': torch.Size([0, 0]),
            'status': 'EMPTY_RANGE',
            'attention_value': None
        }

    # Extract and average attention weights
    sentence_pair_attn = head_attn[start_i:end_i, start_j:end_j]
    avg_attn = sentence_pair_attn.mean()
    
    return {
        'head': head,
        'sentence_i': i,
        'sentence_j': j,
        'range_i': (start_i, end_i),
        'range_j': (start_j, end_j),
        'attention_shape': sentence_pair_attn.shape,
        'status': 'NaN' if torch.isnan(avg_attn) else 'VALID',
        'attention_value': avg_attn.item() if not torch.isnan(avg_attn) else None
    }

def print_attention_pair_details(pair, sentences):
    """Print detailed information about an attention pair.
    """
    print(f"\n[{pair['status']}] Head {pair['head']}, Sentences {pair['sentence_i']}->{pair['sentence_j']}")
    print(f"Range i: {pair['range_i']}, Range j: {pair['range_j']}")
    print(f"Attention submatrix shape: {pair['attention_shape']}")
    if pair['attention_value'] is not None:
        print(f"Attention value: {pair['attention_value']:.4f}")
    print(f"Source sentence: {sentences[pair['sentence_i']]}")
    print(f"Target sentence: {sentences[pair['sentence_j']]}")

def print_aggregate_statistics(sentence_attn):
    """Print aggregate statistics for the entire layer's attention values.
    """
    # Get valid attention values (non-NaN)
    valid_attention = sentence_attn[~torch.isnan(sentence_attn)]
    
    if len(valid_attention) > 0:
        print("\nAggregate Layer Statistics:")
        print(f"Mean attention: {valid_attention.mean().item():.4f}")
        print(f"Median attention: {valid_attention.median().item():.4f}")
        print(f"Min attention: {valid_attention.min().item():.4f}")
        print(f"Max attention: {valid_attention.max().item():.4f}")
        print(f"Standard deviation: {valid_attention.std().item():.4f}")
        
        # Calculate percentiles
        percentiles = torch.tensor([25, 75], dtype=torch.float32)
        quartiles = torch.quantile(valid_attention, percentiles/100)
        print(f"25th percentile: {quartiles[0].item():.4f}")
        print(f"75th percentile: {quartiles[1].item():.4f}")
    else:
        print("\nWarning: No valid attention values found for aggregate statistics!")

def print_layer_summary(layer_idx, all_pairs):
    """Print summary statistics for the layer analysis.
    
    Args:
        layer_idx: Index of the analyzed layer
        all_pairs: List of all analyzed attention pairs
    """
    total_nan_count = len([p for p in all_pairs if p['status'] == 'NaN'])
    valid_count = len([p for p in all_pairs if p['status'] == 'VALID'])
    empty_count = len([p for p in all_pairs if p['status'] == 'EMPTY_RANGE'])
    
    print(f"\nLayer {layer_idx} Summary:")
    print(f"Total pairs analyzed: {len(all_pairs)}")
    print(f"Valid pairs: {valid_count} ({valid_count/len(all_pairs)*100:.1f}%)")
    print(f"NaN values: {total_nan_count} ({total_nan_count/len(all_pairs)*100:.1f}%)")
    print(f"Empty ranges: {empty_count} ({empty_count/len(all_pairs)*100:.1f}%)")

def print_detailed_analysis(sorted_pairs, sentences, num_examples=3):
    """Print detailed analysis for a specified number of pairs from each category.
    """
    # Split pairs by status
    valid_pairs = [p for p in sorted_pairs if p['status'] == 'VALID']
    nan_pairs = [p for p in sorted_pairs if p['status'] == 'NaN']
    empty_pairs = [p for p in sorted_pairs if p['status'] == 'EMPTY_RANGE']
    
    print(f"\nDetailed Analysis (showing up to {num_examples} examples per category):")
    
    if valid_pairs:
        # Group valid pairs by head
        head_pairs = {}
        for pair in valid_pairs:
            head = pair['head']
            if head not in head_pairs:
                head_pairs[head] = []
            head_pairs[head].append(pair)
        
        # Sort pairs within each head by attention value
        for head in head_pairs:
            head_pairs[head].sort(key=lambda x: x['attention_value'], reverse=True)
        
        print("\n=== Strongest Valid Attention Pairs (By Head) ===")
        for head in sorted(head_pairs.keys()):
            head_valid_pairs = head_pairs[head]
            if len(head_valid_pairs) >= 1:
                print(f"\n--- Head {head} ---")
                print("Strongest connection:")
                print_attention_pair_details(head_valid_pairs[0], sentences)
                
                if len(head_valid_pairs) >= 2:
                    print("\nSecond strongest connection:")
                    print_attention_pair_details(head_valid_pairs[1], sentences)
    
    if nan_pairs:
        print("\n=== Sample NaN Attention Pairs ===")
        for pair in nan_pairs[:num_examples]:
            print_attention_pair_details(pair, sentences)
    
    if empty_pairs:
        print("\n=== Sample Empty Range Pairs ===")
        for pair in empty_pairs[:num_examples]:
            print_attention_pair_details(pair, sentences)

def print_top_head_attention_patterns(head_attn, chunk_ranges, sentences, head_info):
    """Print detailed attention patterns for a top head.
    
    Args:
        head_attn: Attention weights for the head
        chunk_ranges: Token ranges for each chunk/sentence
        sentences: List of sentences
        head_info: Tuple of (kurtosis, layer_idx, head_idx)
    """
    kurt, layer, head = head_info
    print(f"\n=== Head {head} (Layer {layer}, Kurtosis: {kurt:.4f}) ===")
    
    # Create sentence-level attention matrix
    n_sentences = len(sentences)
    sentence_attn = np.zeros((n_sentences, n_sentences))
    for i, (start_i, end_i) in enumerate(chunk_ranges):
        for j, (start_j, end_j) in enumerate(chunk_ranges):
            if start_i >= end_i or start_j >= end_j:
                continue
            region = head_attn[start_i:end_i, start_j:end_j]
            if region.size > 0:
                sentence_attn[i, j] = region.mean()
    
    # Find top attention pairs
    pairs = []
    for i in range(n_sentences):
        for j in range(n_sentences):
            if i != j:  # Exclude self-attention
                pairs.append((i, j, sentence_attn[i, j]))
    
    # Sort by attention strength
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Print top 3 strongest attention pairs
    print("\nTop attention patterns:")
    for i, j, attn_value in pairs[:3]:
        print(f"\nAttention strength: {attn_value:.4f}")
        print(f"From sentence [{i}]: {sentences[i]}")
        print(f"To sentence [{j}]: {sentences[j]}")

def analyze_layer(model, tokenizer, prompt, layer_idx=0, verbose=False, num_detailed_examples=3, min_sentence_distance=MIN_SENTENCE_DISTANCE):
    """Analyze attention patterns for a single layer.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text prompt to analyze
        layer_idx: Index of the layer to analyze
        verbose: Whether to print additional debug information
        num_detailed_examples: Number of examples to show for each category
        min_sentence_distance: Minimum number of sentences to ignore for proximity (default: 4)
    
    Returns:
        tuple: (sentence_attn, sentences, all_pairs, kurtosis_list)
    """
    if verbose:
        print(f"\nAnalyzing layer {layer_idx}")
        print(f"Using minimum sentence distance: {min_sentence_distance}")
    
    # Generate text and get attention patterns
    generated_ids, attn_weights, text = generate_text_and_get_attention(model, tokenizer, prompt)
    
    # Process text into sentences and get token ranges
    sentences = split_solution_into_chunks(text)
    if verbose:
        print("\nGenerated sentences:", sentences)
    
    chunk_ranges = get_chunk_ranges(text, sentences)
    if verbose:
        print("\nToken ranges for each sentence:", chunk_ranges)

    # Initialize variables
    num_heads = attn_weights[0].shape[1]
    num_sentences = len(sentences)
    sentence_attn = torch.zeros(num_heads, num_sentences, num_sentences)
    all_pairs = []

    # Calculate kurtosis for each head
    kurtosis_list = []  # List of (kurtosis, layer_idx, head_idx)
    layer_attn = attn_weights[layer_idx][0]  # Remove batch dimension
    
    for head in range(num_heads):
        head_attn = layer_attn[head].cpu().numpy()
        # Calculate kurtosis using the same method as HI_copy_interface
        avg_mat = avg_matrix_by_chunk(head_attn, chunk_ranges)
        vert_scores = get_attn_vert_scores(avg_mat, proximity_ignore=min_sentence_distance, drop_first=0)
        kurt = stats.kurtosis(vert_scores, fisher=True, bias=True, nan_policy="omit")
        kurtosis_list.append((kurt, layer_idx, head))
        
        # Analyze each pair of sentences
        for i, (start_i, end_i) in enumerate(chunk_ranges):
            for j, (start_j, end_j) in enumerate(chunk_ranges):
                pair_info = analyze_sentence_pair_attention(
                    torch.tensor(head_attn), start_i, end_i, start_j, end_j, head, i, j
                )
                all_pairs.append(pair_info)
                
                # Update sentence attention matrix
                if pair_info['attention_value'] is not None:
                    sentence_attn[head, i, j] = pair_info['attention_value']

    # Sort kurtosis list by kurtosis value (descending)
    kurtosis_list.sort(reverse=True, key=lambda x: x[0])
    
    # Print kurtosis analysis and detailed attention patterns for top heads
    print("\nAnalyzing top heads by kurtosis:")
    print("\nAll sentences with indices:")
    for idx, sentence in enumerate(sentences):
        print(f"[{idx}] {sentence}")
        
    print("\nDetailed attention patterns for top heads:")
    for head_info in kurtosis_list[:3]:
        head_idx = head_info[2]
        head_attn = layer_attn[head_idx].cpu().numpy()
        print_top_head_attention_patterns(head_attn, chunk_ranges, sentences, head_info)

    # Print analysis
    print_layer_summary(layer_idx, all_pairs)
    print_aggregate_statistics(sentence_attn)
    
    # Sort and print detailed analysis
    sorted_pairs = sorted(
        all_pairs,
        key=lambda x: (
            0 if x['status'] == 'VALID' else (1 if x['status'] == 'NaN' else 2),
            -x['attention_value'] if x['attention_value'] is not None else 0
        )
    )
    
    # print_detailed_analysis(sorted_pairs, sentences, num_detailed_examples)

    return sentence_attn, sentences, all_pairs, kurtosis_list

def main():
    """Main function to run the attention analysis."""
    # Configuration
    default_prompt = "Should I eat eggs or beef if I care about suffering?"
    layer_to_analyze = 10
    verbose = True  # Changed to True to see min_sentence_distance in output
    num_examples = 3  # Number of examples to show for each category
    min_sentence_distance = MIN_SENTENCE_DISTANCE  # Can be modified here

    # You can modify these parameters here
    prompt = default_prompt  # Change this to analyze different prompts
    
    # Load model and tokenizer - use the preloaded model if available
    preloaded_model = globals().get('model')
    preloaded_tokenizer = globals().get('tokenizer')
    model, tokenizer = get_model_and_tokenizer(preloaded_model, preloaded_tokenizer)

    # Analyze a specific layer
    sentence_attn, sentences, all_pairs, kurtosis_list = analyze_layer(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        layer_idx=layer_to_analyze,
        verbose=verbose,
        num_detailed_examples=num_examples,
        min_sentence_distance=min_sentence_distance
    )
    return sentence_attn, sentences, all_pairs, kurtosis_list

if __name__ == "__main__":
    main()