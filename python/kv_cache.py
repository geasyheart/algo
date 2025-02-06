import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, key_cache=None, value_cache=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Linear transformations
        Q = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Cache mechanism
        if key_cache is not None and value_cache is not None:
            K = torch.cat((key_cache, K), dim=2)
            V = torch.cat((value_cache, V), dim=2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Context vector
        context = torch.matmul(attention, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return context, K, V


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([MultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, key_cache=None, value_cache=None):
        for i, layer in enumerate(self.layers):
            query, key_cache[i], value_cache[i] = layer(query, key, value, key_cache[i] if key_cache else None,
                                                        value_cache[i] if value_cache else None)
        return self.norm(query), key_cache, value_cache


# Example usage
embed_dim = 32
num_heads = 8
num_layers = 6
batch_size = 1
seq_len = 10

# Initialize model
decoder = TransformerDecoder(embed_dim, num_heads, num_layers)

# Dummy input
query = torch.randn(batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)

# Initialize caches
key_cache = [None] * num_layers
value_cache = [None] * num_layers

# First forward pass
output, key_cache, value_cache = decoder(query, key, value, key_cache, value_cache)

# Generate new token
new_query = torch.randn(batch_size, 1, embed_dim)
new_key = torch.randn(batch_size, 1, embed_dim)
new_value = torch.randn(batch_size, 1, embed_dim)

# Second forward pass with cached KV
new_output, key_cache, value_cache = decoder(new_query, new_key, new_value, key_cache, value_cache)

print("Output shape:", output.shape)
print("New output shape:", new_output.shape)
