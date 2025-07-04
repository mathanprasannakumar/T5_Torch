import torch
from torch import nn
import torch.nn.functional as F

import math

from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d



def load_pretrained_weights(t5model_weights, pretrained_model_weights):
    t5model_weights['encoder.token_emb.weight'] =  pretrained_model_weights['encoder.embed_tokens.weight']

    # enc_depth = 8
    for i in range(8):
        if(i == 0):
            t5model_weights['encoder.layer.0.0.fn.fn.relative_position_bias.relative_attention_bias.weight'] \
            = pretrained_model_weights['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']

        t5model_weights[f'encoder.layer.{i}.0.fn.norm.weight'] = pretrained_model_weights[f'encoder.block.{i}.layer.0.layer_norm.weight']
        t5model_weights[f'encoder.layer.{i}.0.fn.fn.to_q.weight'] = pretrained_model_weights[f'encoder.block.{i}.layer.0.SelfAttention.q.weight']
        t5model_weights[f'encoder.layer.{i}.0.fn.fn.to_k.weight'] = pretrained_model_weights[f'encoder.block.{i}.layer.0.SelfAttention.k.weight']
        t5model_weights[f'encoder.layer.{i}.0.fn.fn.to_v.weight'] = pretrained_model_weights[f'encoder.block.{i}.layer.0.SelfAttention.v.weight']
        t5model_weights[f'encoder.layer.{i}.0.fn.fn.to_out.weight'] = pretrained_model_weights[f'encoder.block.{i}.layer.0.SelfAttention.o.weight']

        t5model_weights[f'encoder.layer.{i}.1.fn.norm.weight'] = pretrained_model_weights[f'encoder.block.{i}.layer.1.layer_norm.weight']
        t5model_weights[f'encoder.layer.{i}.1.fn.fn.wi_0.weight'] = pretrained_model_weights[f'encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight']
        t5model_weights[f'encoder.layer.{i}.1.fn.fn.wi_1.weight'] = pretrained_model_weights[f'encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight']
        t5model_weights[f'encoder.layer.{i}.1.fn.fn.wo.weight'] = pretrained_model_weights[f'encoder.block.{i}.layer.1.DenseReluDense.wo.weight']

    # dec_depth = 8 
    for i in range(8):
        if(i == 0):
            t5model_weights['decoder.layer.0.0.fn.fn.relative_position_bias.relative_attention_bias.weight'] \
                = pretrained_model_weights['decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']

        # Self attention
        t5model_weights[f'decoder.layer.{i}.0.fn.norm.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.0.layer_norm.weight']
        t5model_weights[f'decoder.layer.{i}.0.fn.fn.to_q.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.0.SelfAttention.q.weight']
        t5model_weights[f'decoder.layer.{i}.0.fn.fn.to_k.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.0.SelfAttention.k.weight']
        t5model_weights[f'decoder.layer.{i}.0.fn.fn.to_v.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.0.SelfAttention.v.weight']
        t5model_weights[f'decoder.layer.{i}.0.fn.fn.to_out.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.0.SelfAttention.o.weight']

        # Cross attention
        t5model_weights[f'decoder.layer.{i}.1.fn.norm.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.1.layer_norm.weight']
        t5model_weights[f'decoder.layer.{i}.1.fn.fn.to_q.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.1.EncDecAttention.q.weight']
        t5model_weights[f'decoder.layer.{i}.1.fn.fn.to_k.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.1.EncDecAttention.k.weight']
        t5model_weights[f'decoder.layer.{i}.1.fn.fn.to_v.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.1.EncDecAttention.v.weight']
        t5model_weights[f'decoder.layer.{i}.1.fn.fn.to_out.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.1.EncDecAttention.o.weight']

        # Feed forward 
        t5model_weights[f'decoder.layer.{i}.2.fn.norm.weight'] =  pretrained_model_weights[f'decoder.block.{i}.layer.2.layer_norm.weight']
        t5model_weights[f'decoder.layer.{i}.2.fn.fn.wi_0.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight']
        t5model_weights[f'decoder.layer.{i}.2.fn.fn.wi_1.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight']
        t5model_weights[f'decoder.layer.{i}.2.fn.fn.wo.weight'] = pretrained_model_weights[f'decoder.block.{i}.layer.2.DenseReluDense.wo.weight']
        

    t5model_weights[f'encoder.final_norm.weight'] =  \
            pretrained_model_weights[f'encoder.final_layer_norm.weight']

    t5model_weights[f'decoder.final_norm.weight'] =  \
            pretrained_model_weights[f'decoder.final_layer_norm.weight']
        
    t5model_weights[f'to_logits.weight'] =  \
            pretrained_model_weights[f'lm_head.weight']

    return t5model_weights

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://huggingface.co/papers/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = T5LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward layer
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.wi_0 = nn.Linear(dim,inner_dim,bias=False)
        self.wi_1 = nn.Linear(dim,inner_dim,bias=False)
        self.wo = nn.Linear(inner_dim,dim,bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        hidden_relu = self.act(self.wi_0(x))
        hidden_linear = self.wi_1(x)
        x = hidden_relu * hidden_linear 
        x = self.dropout(x)
        x = self.wo(x)
        return x


# T5 relative positional bias
class T5RelativePositionBias(nn.Module):
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 12):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(j - i, j, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None] # this returns the relative postion of each token corresponding to other tokens
        rp_bucket = self._relative_position_bucket(
            rel_pos, 
            causal = self.causal, 
            num_buckets = self.num_buckets, 
            max_distance = self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return qk_dots + (bias * self.scale)

# T5 Self Attention

class T5SelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 12,
        dim_head = 64,
        causal = False,
        dropout = 0.0,
        relative_position_bias = False
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim,bias=False)

        self.rel_pos_bias = relative_position_bias

        if(relative_position_bias):
            self.relative_position_bias = T5RelativePositionBias(scale = dim_head ** -0.5, causal = causal,heads = heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) # (b, h, n, n)

        if(self.rel_pos_bias):
            sim = self.relative_position_bias(sim)

        # mask (b, n)

        mask_value = -torch.finfo(sim.dtype).max

        if mask is not None:
            sim = sim.masked_fill_(~(mask[:, None, :, None].bool()), mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        # attention
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # combine heads and linear output
        return self.to_out(out)

# T5 Cross Attention

class T5CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        context_dim = None,
        heads = 12,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim,bias=False)

        # self.relative_position_bias = T5RelativePositionBias(
        #     scale = dim_head ** -0.5,
        #     causal = False,
        #     heads = heads
        #     )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, mask = None, context_mask = None):
        b, n, _, h = *x.shape, self.heads

        kv_input = default(context, x)

        q, k, v = self.to_q(x), self.to_k(kv_input), self.to_v(kv_input)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) # (b, h, n, n)

        #sim = self.relative_position_bias(sim)

        # mask (b, n)

        mask_value = -torch.finfo(sim.dtype).max

        if mask is not None:
            sim = sim.masked_fill_(~(mask[:, None, :, None].bool()), mask_value)

        if context_mask is not None:
            sim = sim.masked_fill_(~(context_mask[:, None, None, :].bool()), mask_value)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # combine heads and linear output

        return self.to_out(out)

# T5 Encoder

class T5Encoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        #max_seq_len,
        depth,
        heads = 12,
        dim_head = 64,
        causal = True,
        mlp_mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        #self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layer = nn.ModuleList([])
        for i in range(depth): self.layer.append(nn.ModuleList([
                Residual(PreNorm(dim, T5SelfAttention(dim = dim, heads = heads, dim_head = dim_head, causal = causal, dropout = dropout,relative_position_bias=bool(i==0)))),
                Residual(PreNorm(dim, FeedForward(dim = dim, mult = mlp_mult, dropout = dropout))),
            ]))
        self.dropout = nn.Dropout(dropout)
        self.final_norm = T5LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.token_emb(x)
        x = self.dropout(x)

        for attn, mlp in self.layer:
            x = attn(x, mask = mask)
            x = mlp(x)

        x = self.final_norm(x)
        x = self.dropout(x)
        return x

# T5 Decoder

class T5Decoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        #max_seq_len,
        depth,
        heads = 12,
        dim_head = 64,
        causal = True,
        mlp_mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        #self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layer = nn.ModuleList([])
        for i in range(depth):
            self.layer.append(nn.ModuleList([
                Residual(PreNorm(dim, T5SelfAttention(dim = dim, heads = heads, dim_head = dim_head, causal = causal, dropout = dropout,relative_position_bias=bool(i==0)))),
                Residual(PreNorm(dim, T5CrossAttention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim = dim, mult = mlp_mult, dropout = dropout))),
            ]))

        self.final_norm = T5LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, mask = None, context_mask = None):
        x = self.token_emb(x)
        x = self.dropout(x)
        #x = x + self.pos_emb(torch.arange(x.shape[1], device = x.device))

        for attn, cross_attn, mlp in self.layer:
            x = attn(x, mask = mask)
            x = cross_attn(x, context = context, mask = mask, context_mask = context_mask)
            x = mlp(x)

        x = self.final_norm(x)
        x = self.dropout(x)
        return x

# T5

class T5(nn.Module):
    def __init__(
        self,
        *,
        dim,
        #max_seq_len,
        enc_num_tokens,
        enc_depth,
        enc_heads,
        enc_dim_head,
        enc_mlp_mult,
        dec_num_tokens,
        dec_depth,
        dec_heads,
        dec_dim_head,
        dec_mlp_mult,
        dropout = 0.,
        tie_token_emb = True
    ):
        super().__init__()
        
        #self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.encoder = T5Encoder(
            dim = dim,
            #max_seq_len = max_seq_len, 
            num_tokens = enc_num_tokens, 
            depth = enc_depth, 
            heads = enc_heads, 
            dim_head = enc_dim_head, 
            mlp_mult = enc_mlp_mult, 
            dropout = dropout
        )
        
        self.decoder = T5Decoder(
            dim = dim,
            #max_seq_len= max_seq_len, 
            num_tokens = dec_num_tokens, 
            depth = dec_depth, 
            heads = dec_heads, 
            dim_head = dec_dim_head, 
            mlp_mult = dec_mlp_mult, 
            dropout = dropout
        )

        self.to_logits = nn.Linear(dim, dec_num_tokens,bias=False)

        self.decoder_start_token = 0
        self.pad_token_id = 0

        # tie weights
        if tie_token_emb:
            self.encoder.token_emb.weight = self.decoder.token_emb.weight
    
    def shift_right(self,ids):
        shifted_ids = torch.zeros_like(ids)
        shifted_ids[...,1:] = ids[...,:-1]
        shifted_ids[...,0] = self.decoder_start_token
        shifted_ids.masked_fill_(shifted_ids == -100,self.pad_token_id) 

        decoder_attention_mask = (shifted_ids != self.pad_token_id).long()
        decoder_attention_mask[...,0] = 1
        return shifted_ids,decoder_attention_mask

    def generate(self, src, src_mask=None, max_length=50, eos_token_id=1):
        # Step 1: encode input
        encoder_output = self.encoder(src, mask=src_mask)

        b,_ = src.shape
        device = src.device

        # Step 2: start with <pad> or <bos> token
        generated = torch.full((b, 1), 0, dtype=torch.long, device=device)  # assuming 0 is <pad>/<bos>

        for _ in range(max_length):
            # Step 3: decoder forward pass
            x = self.decoder(generated, encoder_output, context_mask=src_mask)

            # Step 4: get next token logits
            logits = self.to_logits(x)  # shape: (b, t, vocab_size)
            next_token_logits = logits[:, -1, :]  # only the last token's logits

            # Step 5: greedy decoding (argmax)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # shape: (b, 1)

            # Step 6: append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Step 7: stop if eos_token is predicted in all sequences
            if (next_token == eos_token_id).all():
                break

        return generated[:, 1:]  # remove the initial start token
    

    def forward(self, src, tgt=None, src_mask = None, tgt_mask = None):

        x = self.encoder(src, mask = src_mask)

        b,t = src.shape

        if(tgt is None):
            return self.generate(src,src_mask)
        else:
            shifted_targets ,decoder_attention_mask= self.shift_right(tgt)
            x = self.decoder(shifted_targets, x, mask = decoder_attention_mask, context_mask = src_mask)

        x = self.to_logits(x)

        loss = F.cross_entropy(x.view(-1,x.shape[-1]),tgt.view(-1),ignore_index=-100)
        return x,loss




