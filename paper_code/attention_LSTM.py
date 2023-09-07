import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        h = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), dim=2)))
        attention = torch.softmax(self.v(energy).squeeze(2), dim=1)
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs)
        return context

class SiameseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(input_size)
        self.attention = Attention(hidden_size)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
#             torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            nn.Linear(hidden_size, output_size))
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
#             torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            nn.Linear(hidden_size, output_size))
#         self.fc1 = nn.Sequential(
#             torch.nn.ReLU(),
#             nn.Linear(hidden_size, output_size))
#         self.fc2 = nn.Sequential(
#             torch.nn.ReLU(),
#             nn.Linear(hidden_size, output_size))

    def forward(self, input1, input2):
        input1 = self.layer_norm(input1)
        input2 = self.layer_norm(input2)
        output_1, (h_n_1, c_n_1) = self.lstm(input1)
        output_2, (h_n_2, c_n_2) = self.lstm(input2)
        
#          # 对LSTM层输出的所有时间步进行归一化
#         batch_size_1, seq_len_1, hidden_size_1 = output_1.size()
#         batch_size_2, seq_len_2, hidden_size_2 = output_2.size()
#         output_1 = self.layer_norm(output_1.view(batch_size_1 * seq_len_1, hidden_size_1)).view(batch_size_1, seq_len_1, hidden_size_1)
#         output_2 = self.layer_norm(output_2.view(batch_size_2 * seq_len_2, hidden_size_2)).view(batch_size_2, seq_len_2, hidden_size_2)

        context_1 = self.attention(h_n_1[-1].unsqueeze(0), output_2)
        context_2 = self.attention(h_n_2[-1].unsqueeze(0), output_1)
        
#         context_1 = self.layer_norm(context_1)
#         context_2 = self.layer_norm(context_2)
#         output_1 = torch.sigmoid(self.fc(context_1.squeeze(1)))
#         output_2 = torch.sigmoid(self.fc(context_2.squeeze(1)))
        output_1 = self.fc1(context_1.squeeze(1))
        output_2 = self.fc2(context_2.squeeze(1))

        return output_1, output_2