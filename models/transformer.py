import torch
import torch.nn as nn
import math
import sys
from models.layers.InceptionTimeBlock import *

# Code is based off stuff taken from the following link:
# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer in the transformer.

    Args:
        - d_model: size of linear layers used
        - num_heads: Number of heads used in multi-head attention
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class FFTransformerLayer(nn.Module):
    """
    Feedforward layer in the transformer. Consists of two fully connected layers
    with a ReLU activation in between. Input and output dimension is d_model.

    Args:
        - d_ff: Number of nodes in the first layer
        - d_model: Number of nodes in the second layer
    """
    def __init__(self, d_model, d_ff):
        super(FFTransformerLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class FFOutputLayer(nn.Module):
    """
    Feedforward layer to process encoder output. Outputs a vector of length 3 corresponding to a probability
    distribution over the possible outcomes of win, lose, draw. To be precise:
        - (1, 0, 0) is a win
        - (0, 1, 0) is a loss
        - (0, 0, 1) is a draw
    Args:
        - d_enc: Dimension of encoder output
        - d_h: Dimension of hidden layer
    """
    def __init__(self, d_enc, d_h):
        super(FFOutputLayer, self).__init__()
        self.fc1 = nn.Linear(d_enc, d_h)
        self.fc2 = nn.Linear(d_h, 3)

    def forward(self, x):
        return torch.softmax(self.fc2(self.fc1(x)), 2)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FFTransformerLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_timespan):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_timespan, d_model)
        position = torch.arange(0, max_timespan, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# note: we have only kept the encoder portion of the transformer for the purpose of prediction making
# the decoder has been discarded, and the encoder's output is given to a feedforward neural network
# in order to make classification predictions


class Transformer(nn.Module):
    """
    Transformer used for prediction.

    Args:
        - d_model: Size of inception block output
        - num_heads: Number of heads used in self-attention
        - num_layers: Number of layers used in the encoder's fully connected layer
        - ver: Hyperparameter which changes the behaviour of the model.
            If ver = 1, we add a single dummy feature to team A + team B features.
            If ver = 2, each team is given its own dummy feature (+2 total).
            If ver = 3, transformer uses no dummy features and just takes in team A + team B features as input.
          The additional feature is added after inception.
    """
    def __init__(self, in_channels, out_channels, num_heads, num_layers, d_ff, d_h, max_timespan, dropout, ver):
        super(Transformer, self).__init__()
        d_model = 4 * out_channels  # note: inception block has output size 4 * out_channels so d_model must be this size
        self.positional_encoding = PositionalEncoding(d_model, max_timespan)
        self.inception = InceptionTimeBlock(in_channels, out_channels)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc1 = nn.Linear(d_model, d_h)
        self.fc2 = nn.Linear(d_h, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.inception(src)
        src_encoded = self.dropout(self.positional_encoding(src))

        enc_output = src_encoded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)
        enc_output = torch.sum(enc_output, 1)
        enc_output = self.fc2(self.fc1(enc_output))  # encoder output fed to fully connected layer
        z = torch.softmax(enc_output, -1)  # softmax to normalize output of fc layer
        batch_size = z.size(0)  # note batch_size is always even
        p1 = z[:(batch_size // 2), :]
        p2 = z[(batch_size // 2):, :]
        p2 = torch.index_select(p2, 1, torch.LongTensor([1, 0, 2]))
        return (p1 + p2) / 2

######

# Note: All code below is for training. I originally put this in train.py so the data isn't loaded here,
# move the code if you need to or just load the data in here

#####

in_channels = 14  # this should be the number of features in each data sample
# the remaining inputs are hyperparameters which can be tuned accordingly
out_channels = 256
num_heads = 8  # note: num heads must divide d_model
num_layers = 6
d_ff = 256
d_h = 512
max_timespan = 50
dropout = 0.1


def train_transformer():
    transformer_model = Transformer(in_channels, out_channels, num_heads, num_layers, d_ff, d_h, max_timespan, dropout, 3)
    transformer_model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    iter = 0
    for epoch in range(20):
        for home_teams, away_teams, home_features, away_features, targets in train_loader:
            optimizer.zero_grad()
            concat_features = torch.cat((home_features, away_features), 0)
            output = transformer_model(concat_features)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            pred = torch.argmax(output, 1)
            acc = (pred == targets).float().mean()
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Accuracy: {acc}")
            iter = iter + 1
            if iter > 200:
                break
        break

    for epoch in range(20):
        for home_teams, away_teams, home_features, away_features, targets in val_loader:
            concat_features = torch.cat((home_features, away_features), 0)
            output = transformer_model(concat_features)
            pred = torch.argmax(output, 1)
            acc = (pred == targets).float().mean()
            print(f"Accuracy: {acc}")

