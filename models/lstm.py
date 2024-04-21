import torch
import torch.nn as nn

from models.InceptionTime import InceptionTime
from models.utils import get_last_inception_output_size

class LSTM(nn.Module):
    def __init__(self, num_features, inception_depth, inception_out, hidden_size, num_heads, num_classes, bottleneck_dim = None):
        super(LSTM, self).__init__()

        self.num_features   = num_features

        self.inception_depth= inception_depth
        self.inception_out  = inception_out

        self.hidden_size    = hidden_size

        self.num_heads      = num_heads
        self.num_classes    = num_classes

        self.inception = InceptionTime(num_features, inception_out, depth=inception_depth, bottleneck_dim=bottleneck_dim)

        self.lstm = nn.LSTM(input_size= get_last_inception_output_size(inception_out, inception_depth), hidden_size=hidden_size, batch_first=True)

        self.attn_H = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.attn_A = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)

        self.fc = nn.Linear(2 * hidden_size, num_classes)


    def forward(self, home, away):

        batch_size = home.size(0)
        seq_length = home.size(1)

        stacked_x = torch.stack((home, away)).reshape(2 * batch_size, seq_length, self.num_features)

        inception_features = self.inception(stacked_x)

        indices_H = torch.arange(0, batch_size)
        indices_A = torch.arange(batch_size, 2 * batch_size)

        seq_H = torch.index_select(inception_features, 0, indices_H) # (batch_size, seq_length, num_features)
        seq_A = torch.index_select(inception_features, 0, indices_A)

        h_H, _ = self.lstm(seq_H) # (batch_size, seq_length, hidden_size)
        h_A, _ = self.lstm(seq_A) # same ^

        self_attn_H, _ = self.attn_H(h_H, h_H, h_H)
        self_attn_A, _ = self.attn_A(h_A, h_A, h_A)

        code = torch.cat([self_attn_H[:,-1,:], self_attn_A[:,-1,:]], dim=-1) #  (batch_size, 2 * (hidden_size + 1))

        z = nn.functional.softmax(self.fc(code), dim=-1)

        return z
