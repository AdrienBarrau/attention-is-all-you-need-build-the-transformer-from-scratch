"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    words=[word for sentence in sentences for word in sentence.split()]
    m=len(specials)
    n=len(words)
    dico={specials[i]:i for i in range(m)}
    j=0
    i=0
    while (j<n):
        if words[j] not in dico:
            dico[words[j]]=i+m
            i+=1
        j+=1
    
    return dico

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    
    return {token_id: token for token, token_id in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    res=[]
    words=sentence.split()
    n=len(words)
    for i in range (n):
        if words[i] not in token_to_id.keys():
            res.append(token_to_id[unk_token])
        else:
            res.append(token_to_id[words[i]])
    return res

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    n=len(ids)
    for i in range(n):
        ids[i]=id_to_token[ids[i]]
    return ids

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    n=len(ids)
    if n>max_len:
        return ids[:max_len]
    elif n<max_len:
        ids=ids+[pad_id for i in range (max_len-n)]
        return ids
    else:
        return ids

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    res=torch.tensor([liste for liste in padded_sequences],dtype=torch.long)
    return res

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch
import math
def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    return torch.tensor([math.exp(2*i*(-math.log(10000)/d_model)) for i in range (d_model//2)])

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    res= torch.tensor([float(i) for i in range (max_len)])
    return res.unsqueeze(-1)

# Step 10 - fill_even_indices_with_sin
import torch
import math

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    n,m=pe.shape
    for i in range(n):
        for j in range(m//2):
            pe[i][2*j]=math.sin(position[i]*div_term[j])
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch
import math
def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    n,m=pe.shape
    for i in range(n):
        for j in range (m//2):
            pe[i][2*j+1]=math.cos(position[i]*div_term[j])
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    div_term=compute_positional_div_term(d_model)
    pos= build_position_index_column(max_len)
    pe=torch.ones(max_len,d_model)
    pe=fill_odd_indices_with_cos(pe,pos,div_term)
    pe=fill_even_indices_with_sin(pe,pos,div_term)
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.

    L = embedded_batch.shape[1]
    return embedded_batch + positional_encoding[:L, :]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    
    mask=token_ids.ne(pad_id)
    return mask.unsqueeze(1).unsqueeze(2)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    mask=torch.ones(seq_len,seq_len,dtype=bool)
    for i in range(seq_len):
        for j in range (seq_len):
            if i<j:
                mask[i][j]=False
            else:
                mask[i][j]=True

    return mask.unsqueeze(0).unsqueeze(0)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    return padding_mask&causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    key_transp=key.transpose(-1,-2)
    return torch.matmul(query,key_transp)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores/math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch
import math
def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill(~mask,-math.inf)

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    weights=torch.softmax(masked_scores,dim=-1)
    weights=torch.nan_to_num(weights,nan=0.0)
    return weights

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return torch.matmul(attention_weights, value)

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    # 1. Compute raw scores (Shape: batch, num_heads, L, L)
    raw_scores = compute_raw_attention_scores(query, key)
    
    # 2. Get d_k (the size of the head dimension) from the last axis of query
    d_k = query.size(-1)
    
    # 3. Scale by sqrt(d_k)
    scaled_scores = raw_scores / math.sqrt(d_k)
    
    # 4. Optionally apply mask
    if mask is not None:
        scaled_scores = mask_attention_scores_with_neg_inf(scaled_scores, mask)
        
    # 5. Turn scores into probabilities (attention weights)
    attention_weights = softmax_attention_weights(scaled_scores)
    
    # 6. Mix values to get the "context" vector
    context = apply_attention_weights_to_values(attention_weights, value)
    
    # Return both the context vector and the weights as requested by the docstring
    return context, attention_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    [B, L, d_model]=list(tensor.size())
    return tensor.reshape(B,L,num_heads,d_model//num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.transpose(1,2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    [B,num_heads,L,d_k]=list(multi_head_tensor.size())
    res=multi_head_tensor.transpose(1,2)

    return res.reshape(B,L,num_heads*d_k)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    if bias is None:
        return torch.matmul(x,weight.T)
    else:
        return (torch.matmul(x,weight.T)+bias)

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    q=apply_linear_projection(x,w_q,b_q)
    k=apply_linear_projection(x,w_k,b_k)
    v=apply_linear_projection(x,w_v,b_v)
    return (q,k,v)

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    [B,L,d_model]=list(q.size())
    d_k=d_model//num_heads
    q_h=(q.reshape(B,-1,num_heads,d_k)).transpose(1,2)
    k_h=(k.reshape(B,-1,num_heads,d_k)).transpose(1,2)
    v_h=(v.reshape(B,-1,num_heads,d_k)).transpose(1,2)
    return (q_h,k_h,v_h)

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    embed=merge_heads_back_to_model_dim(context)
    
    return apply_linear_projection(embed,w_o,b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    q = apply_linear_projection(query, w_q, None)
    k = apply_linear_projection(key, w_k, None)
    v = apply_linear_projection(value, w_v, None)

    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)

    context, weights = multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask)
 
    return merge_heads_and_project_output(context, w_o, None)

# Step 32 - apply_ffn_first_linear_and_relu
import torch
import torch.nn.functional as F


def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    #w1=w1.unsqueeze(0)
    hidden = torch.matmul(x, w1) + b1

    # ReLU: out-of-place — do NOT use .apply_() or .relu_()
    hidden = F.relu(hidden)          # or: torch.relu(hidden)
    # or equivalently: hidden = hidden.clamp(min=0)

    return hidden

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    z=torch.matmul(hidden,w2)+b2
    return z

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    return (apply_ffn_second_linear(apply_ffn_first_linear_and_relu(x,w1,b1),w2,b2))

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean=0
    moment=0
    n=x.shape[-1]
    for i in range(n):
        mean+=x[...,i]
        moment=moment+x[...,i]*x[...,i]
    mean=mean/n
    moment=moment/n 
    variance=moment-mean*mean
    
    return (mean.unsqueeze(-1),variance.unsqueeze(-1))

# Step 36 - normalize_and_scale_with_gamma_beta
import torch
import math
def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean,var=compute_layer_norm_mean_and_variance(x)
    x_norm=(x-mean)/torch.sqrt(eps+var)
    n=x.shape[-1]
    y=gamma*x_norm+beta
    return y

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    return normalize_and_scale_with_gamma_beta(residual_input+sublayer_output,gamma,beta,eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    num_mask=keep_mask.to(x.dtype)
    res=(x*keep_mask)/keep_prob
    return res

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    context=assemble_multi_head_attention_forward(x,x,x,w_q, w_k, w_v, w_o,num_heads, src_mask)
    embedding=apply_residual_add_and_norm(x,context,gamma,beta,eps=1e-5)
    return embedding

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    proj=position_wise_feed_forward_network(x,w1,b1,w2,b2)
    add_norm=apply_residual_add_and_norm(x,proj,gamma,beta,eps=1e-5)
    return add_norm

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    layer_1=encoder_layer_self_attention_sublayer(x,layer_params['w_q'],layer_params['w_k'],layer_params['w_v'],layer_params['w_o'],layer_params['attn_gamma'],layer_params['attn_beta'],num_heads,src_mask)
    layer_2=encoder_layer_feed_forward_sublayer(layer_1,layer_params['w1'],layer_params['b1'],layer_params['w2'],layer_params['b2'],layer_params['ffn_gamma'],layer_params['ffn_beta'])
    return layer_2

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    hidden=x
    for layer_params in encoder_layer_params_list:
        hidden=assemble_encoder_layer(hidden,layer_params, num_heads, src_mask)
    return hidden

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    projection_mha=assemble_multi_head_attention_forward(y,y,y,w_q,w_k,w_v,w_o,num_heads,tgt_mask)
    norm=apply_residual_add_and_norm(y,projection_mha,gamma,beta)
    return norm

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None:
        if src_mask.dim() == 2:
            # (B, L_s) per-example key padding mask -> (B, 1, 1, L_s)
            src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        elif src_mask.dim() == 4 and src_mask.size(-2) == src_mask.size(-1) and src_mask.size(-2) != 1:
            # (B, 1, L_s, L_s) combined padding&causal mask -> drop the causal part.
            # The last query row of a lower-triangular causal mask is all-True,
            # so slicing it leaves the pure padding mask: (B, 1, 1, L_s).
            src_mask = src_mask[..., -1:, :]
        # (else: already (B, 1, 1, L_s) or otherwise broadcastable — pass through)

    proj_mhca = assemble_multi_head_attention_forward(
        y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask
    )
    norm = apply_residual_add_and_norm(y, proj_mhca, gamma, beta)
    return norm

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    hidden=position_wise_feed_forward_network(y, w1, b1, w2, b2)
    return apply_residual_add_and_norm(y,hidden,gamma,beta)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN.

    layer_params keys (all torch tensors):
      masked self-attention : w_q_self, w_k_self, w_v_self, w_o_self, self_gamma, self_beta
      cross-attention       : w_q_cross, w_k_cross, w_v_cross, w_o_cross, cross_gamma, cross_beta
      feed-forward          : w1, b1, w2, b2, ffn_gamma, ffn_beta
    """
    # TODO: chain the three decoder sublayers using params from layer_params.
    [w_q, w_k, w_v, w_o, gamma, beta]=[layer_params['w_q_self'],layer_params['w_k_self'],layer_params['w_v_self'],layer_params['w_o_self'],layer_params['self_gamma'],layer_params['self_beta']]
    hidden_self_attn=decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask)
    [w_q, w_k, w_v, w_o, gamma, beta]=[layer_params['w_q_cross'],layer_params['w_k_cross'],layer_params['w_v_cross'],layer_params['w_o_cross'],layer_params['cross_gamma'],layer_params['cross_beta']]
    hidden_cross_attn=decoder_layer_cross_attention_sublayer(hidden_self_attn, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask)
    [w1, b1, w2, b2, ffn_gamma, ffn_beta]=[layer_params['w1'],layer_params['b1'],layer_params['w2'],layer_params['b2'],layer_params['ffn_gamma'],layer_params['ffn_beta']]
    hidden=decoder_layer_feed_forward_sublayer(hidden_cross_attn, w1, b1, w2, b2, ffn_gamma, ffn_beta)
    return hidden

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
   
    n=len(decoder_layer_params_list)
    hidden=y
    for i in range (n):
        hidden=assemble_decoder_layer(hidden, encoder_output, decoder_layer_params_list[i], num_heads, src_mask, tgt_mask)
    return hidden

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.T

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    probability_vector=torch.log(torch.softmax(logits,dim=2))  #logical, we want a distribution across the vocabulary
    return probability_vector

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.

    # ── 0. Pull out the parameter groups ──
    encoder_layer_params = model_params['encoder_layers']
    decoder_layer_params = model_params['decoder_layers']

    # Token embedding matrix: (vocab_size, d_model). Accept either a raw tensor
    # or a dict with a 'weight' entry, depending on the harness.
    token_embedding = model_params['token_embedding']
    if isinstance(token_embedding, dict):
        token_embedding = token_embedding['weight']

    # Output projection: accept a tensor, a dict {'weight', 'bias'}, or fall
    # back to tying it to the token embedding matrix (step 049).
    output_projection = model_params.get('output_projection')
    if output_projection is None:
        proj_weight = tie_output_projection_to_token_embeddings(token_embedding)
        proj_bias = None
    elif isinstance(output_projection, dict):
        proj_weight = output_projection['weight']
        proj_bias = output_projection.get('bias')
    else:
        proj_weight = output_projection
        proj_bias = None

    # ── 1. Token embedding lookup + sqrt(d_model) scaling (steps 007) ──
    src_embedded = scale_embeddings_by_sqrt_d_model(token_embedding[src_ids], token_embedding.shape[-1])
    tgt_embedded = scale_embeddings_by_sqrt_d_model(token_embedding[tgt_ids], token_embedding.shape[-1])

    # ── 2. Positional encoding: shapes now come from the EMBEDDED tensors ──
    d_model = src_embedded.shape[-1]
    max_src_len = src_embedded.shape[1]
    max_tgt_len = tgt_embedded.shape[1]          # ← from tgt_ids, not src_ids!

    pe = build_sinusoidal_positional_encoding(max(max_src_len, max_tgt_len), d_model)
    src_embedded = add_positional_encoding_to_embeddings(src_embedded, pe)
    tgt_embedded = add_positional_encoding_to_embeddings(tgt_embedded, pe)

    # ── 3. Masks ──
    # Encoder: padding mask ONLY — the source is attended bidirectionally,
    # so no causal mask here.
    src_padding_mask = build_padding_mask(src_ids, pad_id)          # (B, 1, 1, L_src)
    src_mask = src_padding_mask

    # Decoder self-attention: causal AND padding, built from the TARGET ids.
    tgt_padding_mask = build_padding_mask(tgt_ids, pad_id)          # (B, 1, 1, L_tgt)
    tgt_causal_mask = build_causal_mask(max_tgt_len)                # (1, 1, L_tgt, L_tgt)
    tgt_mask = combine_padding_and_causal_masks(tgt_padding_mask, tgt_causal_mask)

    # ── 4. Encoder / decoder stacks ──
    encoder_output = stack_encoder_layers(src_embedded, encoder_layer_params, num_heads, src_mask)
    decoder_output = stack_decoder_layers(tgt_embedded, encoder_output,
                                          decoder_layer_params, num_heads,
                                          src_mask, tgt_mask)

    # ── 5. Output projection to vocab logits, then log-softmax ──
    logits = apply_final_output_projection(decoder_output, proj_weight, proj_bias)
    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    dico={}
    dico['w_q']=(torch.randn((d_model,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['w_k']=(torch.randn((d_model,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['w_v']=(torch.randn((d_model,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['w_o']=(torch.randn((d_model,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    
    dico['w1']=(torch.randn((d_model,d_ff),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['b1']=torch.zeros((d_ff,),dtype=torch.float32,requires_grad=True)
    dico['w2']=(torch.randn((d_ff,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['b2']=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)

    dico['attn_gamma']=torch.ones((d_model,),dtype=torch.float32,requires_grad=True)
    dico['attn_beta']=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)
    dico['ffn_gamma']=torch.ones((d_model,),dtype=torch.float32,requires_grad=True)
    dico['ffn_beta']=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)
    return dico

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    dico={}
    dico['w_q_self']=(torch.randn((d_model,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['w_k_self']=(torch.randn((d_model,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['w_v_self']=(torch.randn((d_model,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['w_o_self']=(torch.randn((d_model,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    
    dico['w_q_cross']=(torch.randn((d_model,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['w_k_cross']=(torch.randn((d_model,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['w_v_cross']=(torch.randn((d_model,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['w_o_cross']=(torch.randn((d_model,d_model),dtype=torch.float32)/10.0).requires_grad_(True)

    dico['w1']=(torch.randn((d_model,d_ff),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['b1']=torch.zeros((d_ff,),dtype=torch.float32,requires_grad=True)
    dico['w2']=(torch.randn((d_ff,d_model),dtype=torch.float32)/10.0).requires_grad_(True)
    dico['b2']=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)

    dico['self_gamma']=torch.ones((d_model,),dtype=torch.float32,requires_grad=True)
    dico['self_beta']=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)
    dico['ffn_gamma']=torch.ones((d_model,),dtype=torch.float32,requires_grad=True)
    dico['ffn_beta']=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)

    dico['cross_gamma']=torch.ones((d_model,),dtype=torch.float32,requires_grad=True)
    dico['cross_beta']=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)
    
    return dico

# Step 54 - init_embedding_and_projection_parameters
import torch
'''
def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    dico = {}

    # src embedding is a separate matrix; tgt embedding and the output
    # projection share one tensor (weight tying).
    dico['src_embedding'] = torch.zeros(
        [vocab_size, d_model], dtype=torch.float32
    ).requires_grad_(True)

    tgt_embed_tensor = torch.zeros(
        [vocab_size, d_model], dtype=torch.float32
    ).requires_grad_(True)
    dico['tgt_embedding'] = tgt_embed_tensor

    if tie_weights:
        dico['output_projection'] = tgt_embed_tensor   # same object as tgt_embedding
    else:
        dico['output_projection'] = torch.zeros(
            [vocab_size, d_model], dtype=torch.float32
        ).requires_grad_(True)

    return dico
'''

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    dico = {}

    def _random_embedding():
        w = torch.randn(vocab_size, d_model, dtype=torch.float32)
        w *= 0.01                       # in-place scale keeps `w` a leaf tensor
        return w.requires_grad_(True)   # safe: w is still a leaf

    dico['src_embedding'] = _random_embedding()

    tgt_embed_tensor = _random_embedding()
    dico['tgt_embedding'] = tgt_embed_tensor

    if tie_weights:
        dico['output_projection'] = tgt_embed_tensor   # same object as tgt_embedding
    else:
        dico['output_projection'] = _random_embedding()

    return dico

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    seen_ids = set()
    flat_params = []

    def add_unique_tensors(iterable_or_dict):
        # Handle dictionary inputs directly
        values = iterable_or_dict.values() if isinstance(iterable_or_dict, dict) else iterable_or_dict
        for tensor in values:
            tensor_id = id(tensor)
            if tensor_id not in seen_ids:
                seen_ids.add(tensor_id)
                flat_params.append(tensor)

    # 1. Process Encoder Layers
    for layer in encoder_layer_params:
        add_unique_tensors(layer)

    # 2. Process Decoder Layers
    for layer in decoder_layer_params:
        add_unique_tensors(layer)

    # 3. Process Embedding Parameters
    # Using a list ensures we evaluate them in a predictable order
    emb_candidates = [
        embedding_params.get('src_embedding'),
        embedding_params.get('tgt_embedding'),
        embedding_params.get('output_projection')
    ]
    # Filter out None values in case any key is missing
    add_unique_tensors([t for t in emb_candidates if t is not None])

    return flat_params

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    res=[]
    for i in range (len(target_ids)):
        n=len(target_ids[i])-1
        res.append([start_token_id]+target_ids[i][:n].tolist())
    return torch.tensor(res)

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    return d_model**(-1/2)*min(step**(-1/2),step*warmup_steps**(-3/2))

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    return torch.ones(shape)*(epsilon / (vocab_size - 2))

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    liste=smoothed_distribution.tolist()
    B,tgt_seq_len,vocab_size=smoothed_distribution.shape
    for i in range(B):
        for j in range(tgt_seq_len):
            liste[i][j][gold_token_ids[i][j]]=confidence
    return torch.tensor(liste)

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
    B,T,C=smoothed_distribution.shape
    liste=smoothed_distribution.tolist()
    for i in range(B):
        for j in range(T):
            if gold_token_ids[i][j]==pad_id:
                liste[i][j]=[0 for i in range (C)]

            liste[i][j][pad_id]=0
    return torch.tensor(liste)

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    loss=0
    B,T,C=log_probabilities.shape
    for i in range (B):
        for j in range (T):
            for k in range(C):
                loss-=log_probabilities[i][j][k]*smoothed_distribution[i][j][k]
    return loss

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids

    B,T=gold_token_ids.shape
    nb_non_pad=0
    for i in range(B):
        for j in range(T):
            if gold_token_ids[i][j]!=pad_id:
                nb_non_pad+=1
    if nb_non_pad==0:
        return total_loss
    else:
        avg_loss=total_loss/nb_non_pad
        return avg_loss

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only
    predicted_token_ids = log_probabilities.argmax(dim=-1)

    # Correctness per position, as float (1.0 if match, else 0.0)
    correct = (predicted_token_ids == gold_token_ids).float()

    # Mask out pad positions: 1.0 for real tokens, 0.0 for pad
    non_pad_mask = (gold_token_ids != pad_id).float()

    # Count only non-pad positions
    num_non_pad = non_pad_mask.sum()

    if num_non_pad == 0:
        return torch.tensor(0.0, device=log_probabilities.device)

    # Sum correct predictions over non-pad positions, divide by count
    accuracy = (correct * non_pad_mask).sum() / num_non_pad

    return accuracy

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    state = {
        'm': [torch.zeros_like(p) for p in parameter_list],
        'v': [torch.zeros_like(p) for p in parameter_list],
        't': 0,
    }
    return state

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    return beta1 * m_prev + (1 - beta1) * grad

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    return beta2 * v_prev + (1 - beta2) * grad ** 2

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    return (m_t/((1 - beta1**step)),v_t/((1 - beta2**step)))

# Step 68 - compute_adam_parameter_update
import torch

def compute_adam_parameter_update(m_hat, v_hat, learning_rate, epsilon):
    """Return delta = learning_rate * m_hat / (sqrt(v_hat) + epsilon); the caller subtracts it."""
    # TODO: compute the Adam step from the bias-corrected moments without tracking gradients
    return learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    # TODO: increment t, then for each param with a grad update m, v, bias-correct, and subtract delta in place.
    optimizer_state['t'] += 1
    t = optimizer_state['t']

    bias_correction1 = 1 - beta1 ** t
    bias_correction2 = 1 - beta2 ** t

    with torch.no_grad():
        for i, param in enumerate(parameter_list):
            grad = param.grad
            if grad is None:
                continue

            m_prev = optimizer_state['m'][i]
            v_prev = optimizer_state['v'][i]

            m_t = beta1 * m_prev + (1 - beta1) * grad
            v_t = beta2 * v_prev + (1 - beta2) * grad ** 2

            optimizer_state['m'][i] = m_t
            optimizer_state['v'][i] = v_t

            m_hat = m_t / bias_correction1
            v_hat = v_t / bias_correction2

            param -= learning_rate * m_hat / (v_hat.sqrt() + epsilon)

    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    for param in parameter_list:
        if param.grad is not None:
            param.grad = None

# Step 71 - compute_batch_training_loss
def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    """One teacher-forced forward pass -> label-smoothed KL loss averaged over non-pad tokens."""
    pad_id     = config['pad_id']
    start_id   = config['start_id']
    vocab_size = config['vocab_size']
    smoothing  = config['smoothing']
    num_heads  = config['num_heads']

    # 1. Teacher-forced decoder input: shift target right with <bos>.
    decoder_input = shift_targets_right_with_start_token(tgt_batch, start_id)

    # 2. One full Transformer forward pass -> (B, T, V) log-probabilities.
    log_probabilities = run_transformer_forward(
        src_batch, decoder_input, model_params, num_heads, pad_id
    )

    # 3. Build the label-smoothed gold distribution (B, T, V) against the unshifted gold.
    smoothed_distribution = build_uniform_smoothing_distribution(
        log_probabilities.shape, vocab_size, smoothing
    )
    smoothed_distribution = set_confidence_on_gold_tokens(
        smoothed_distribution, tgt_batch, confidence=1 - smoothing
    )
    smoothed_distribution = zero_pad_column_and_pad_token_rows(
        smoothed_distribution, tgt_batch, pad_id
    )

    # 4. Summed KL loss over all entries, then average over non-pad tokens.
    total_loss = compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution)
    return average_loss_over_non_pad_tokens(total_loss, tgt_batch, pad_id)

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

