import torch
import torch.nn as nn


class deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim=None):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, device):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class Deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim):
        super(Deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        torch.nn.init.uniform_(self.embedding.weight)
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(self.embedding_dim,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc0 = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, device):
        input0 = features[0]
        embed0 = self.embedding(input0)
        h0 = torch.zeros(self.num_layers, embed0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, embed0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(embed0, (h0, c0))
        out0 = self.fc0(out[:, -1, :])
        return out0


#
# # log key and parameter value as input and output
# class deeplog1(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_keys):
#         super(deeplog1, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.embedding_dim = 50
#         self.embedding_size = num_keys + 1  # +1 for padding log key 0
#         self.parameter_dim = 1
#         self.embedding = nn.Embedding(self.embedding_size, self.embedding_dim)
#         torch.nn.init.uniform_(self.embedding.weight)
#         self.embedding.weight.requires_grad = True
#
#         self.lstm = nn.LSTM(self.embedding_dim + self.parameter_dim,
#                             hidden_size,
#                             num_layers,
#                             batch_first=True)
#         self.fc0 = nn.Linear(hidden_size, self.embedding_size)
#         self.fc1 = nn.Linear(hidden_size, self.parameter_dim)
#
#     def forward(self, features, device):
#         input0, input1 = features
#         embed0 = self.embedding(input0)
#         multi_input = torch.cat((embed0, input1), 2)
#         h0 = torch.zeros(self.num_layers, multi_input.size(0),
#                          self.hidden_size).to(device)
#         c0 = torch.zeros(self.num_layers, multi_input.size(0),
#                          self.hidden_size).to(device)
#         out, _ = self.lstm(multi_input, (h0, c0))
#         out0 = self.fc0(out[:, -1, :])
#         out1 = self.fc1(out[:, -1, :])
#         return out0, out1
#
# #two lstm and merge two hidden state
# class deeplog2(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_keys):
#         super(deeplog2, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.embedding_dim = 50
#         self.embedding_size = num_keys + 1 # +1 for padding log key 0
#         self.embedding = nn.Embedding(self.embedding_size, self.embedding_dim)
#         torch.nn.init.uniform_(self.embedding.weight)
#         self.embedding.weight.requires_grad = True
#
#         self.lstm0 = nn.LSTM(self.embedding_dim,
#                             hidden_size,
#                             num_layers,
#                             batch_first=True)
#         self.lstm1 = nn.LSTM(input_size,
#                             hidden_size,
#                             num_layers,
#                             batch_first=True)
#
#         self.fc0 = nn.Linear(2*hidden_size, num_keys)
#         self.fc1 = nn.Linear(2*hidden_size, 1)  # num of parameters, timestamp
#
#     def forward(self, features, device):
#         input0, input1 = features[0], features[1]
#         embedd0 = self.embedding(input0)
#         h0_0 = torch.zeros(self.num_layers, embedd0.size(0),
#                            self.hidden_size).to(device)
#         c0_0 = torch.zeros(self.num_layers, embedd0.size(0),
#                            self.hidden_size).to(device)
#
#         out0, _ = self.lstm0(embedd0, (h0_0, c0_0))
#
#         h0_1 = torch.zeros(self.num_layers, input1.size(0),
#                            self.hidden_size).to(device)
#         c0_1 = torch.zeros(self.num_layers, input1.size(0),
#                            self.hidden_size).to(device)
#
#         out1, _ = self.lstm1(input1, (h0_1, c0_1))
#         multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
#
#         multi_out0 = self.fc0(multi_out)
#         multi_out1 = self.fc1(multi_out)
#         return multi_out0, multi_out1
#
# # two lstm and not merge hidden state
# class deeplog3(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_keys):
#         super(deeplog3, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.embedding_dim = 50
#         self.embedding_size = num_keys + 1 # +1 for padding log key 0
#         self.embedding = nn.Embedding(self.embedding_size, self.embedding_dim)
#         torch.nn.init.uniform_(self.embedding.weight)
#         self.embedding.weight.requires_grad = True
#
#         self.lstm0 = nn.LSTM(self.embedding_dim,
#                             hidden_size,
#                             num_layers,
#                             batch_first=True)
#         self.lstm1 = nn.LSTM(input_size,
#                             hidden_size,
#                             num_layers,
#                             batch_first=True)
#
#         self.fc0 = nn.Linear(hidden_size, num_keys)
#         self.fc1 = nn.Linear(hidden_size, 1)  # num of parameters, timestamp
#
#     def forward(self, features, device):
#         input0, input1 = features[0], features[1]
#         embedd0 = self.embedding(input0)
#         h0_0 = torch.zeros(self.num_layers, embedd0.size(0),
#                            self.hidden_size).to(device)
#         c0_0 = torch.zeros(self.num_layers, embedd0.size(0),
#                            self.hidden_size).to(device)
#
#         out0, _ = self.lstm0(embedd0, (h0_0, c0_0))
#
#         h0_1 = torch.zeros(self.num_layers, input1.size(0),
#                            self.hidden_size).to(device)
#         c0_1 = torch.zeros(self.num_layers, input1.size(0),
#                            self.hidden_size).to(device)
#
#         out1, _ = self.lstm1(input1, (h0_1, c0_1))
#
#         multi_out0 = self.fc0(out0[:, -1, :])
#         multi_out1 = self.fc1(out1[:, -1, :])
#         return multi_out0, multi_out1
#
#
# class loganomaly0(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_keys):
#         super(loganomaly0, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm0 = nn.LSTM(input_size,
#                              hidden_size,
#                              num_layers,
#                              batch_first=True)
#         self.lstm1 = nn.LSTM(input_size,
#                              hidden_size,
#                              num_layers,
#                              batch_first=True)
#         self.fc = nn.Linear(2 * hidden_size, num_keys)
#         self.attention_size = self.hidden_size
#
#         self.w_omega = Variable(
#             torch.zeros(self.hidden_size, self.attention_size))
#         self.u_omega = Variable(torch.zeros(self.attention_size))
#
#         self.sequence_length = 28
#
#     def attention_net(self, lstm_output):
#         output_reshape = torch.Tensor.reshape(lstm_output,
#                                               [-1, self.hidden_size])
#         attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
#         attn_hidden_layer = torch.mm(
#             attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
#         exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
#                                     [-1, self.sequence_length])
#         alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
#         alphas_reshape = torch.Tensor.reshape(alphas,
#                                               [-1, self.sequence_length, 1])
#         state = lstm_output
#         attn_output = torch.sum(state * alphas_reshape, 1)
#         return attn_output
#
#     def forward(self, features, device):
#         input0, input1 = features[0], features[1]
#         h0_0 = torch.zeros(self.num_layers, input0.size(0),
#                            self.hidden_size).to(device)
#         c0_0 = torch.zeros(self.num_layers, input0.size(0),
#                            self.hidden_size).to(device)
#
#         out0, _ = self.lstm0(input0, (h0_0, c0_0))
#
#         h0_1 = torch.zeros(self.num_layers, input1.size(0),
#                            self.hidden_size).to(device)
#         c0_1 = torch.zeros(self.num_layers, input1.size(0),
#                            self.hidden_size).to(device)
#
#         out1, _ = self.lstm1(input1, (h0_1, c0_1))
#
#         multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
#         out = self.fc(multi_out)
#         return out, out
#
#
# class loganomaly2(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_keys):
#         super(loganomaly2, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#
#         self.lstm0 = nn.LSTM(input_size,
#                             hidden_size,
#                             num_layers,
#                             batch_first=True)
#         self.lstm1 = nn.LSTM(input_size,
#                             hidden_size,
#                             num_layers,
#                             batch_first=True)
#         self.lstm2 = nn.LSTM(input_size,
#                             hidden_size,
#                             num_layers,
#                             batch_first=True)
#
#         self.fc0 = nn.Linear(3 * hidden_size, num_keys)
#         self.fc1 = nn.Linear(3 * hidden_size, 1)
#
#     def forward(self, features, device):
#         input0, input1, input2 = features[0], features[1], features[2]
#         h0_0 = torch.zeros(self.num_layers, input0.size(0),
#                            self.hidden_size).to(device)
#         c0_0 = torch.zeros(self.num_layers, input0.size(0),
#                            self.hidden_size).to(device)
#
#         out0, _ = self.lstm0(input0.float(), (h0_0, c0_0))
#
#         h0_1 = torch.zeros(self.num_layers, input1.size(0),
#                            self.hidden_size).to(device)
#         c0_1 = torch.zeros(self.num_layers, input1.size(0),
#                            self.hidden_size).to(device)
#
#         out1, _ = self.lstm1(input1, (h0_1, c0_1))
#
#         h0_2 = torch.zeros(self.num_layers, input2.size(0),
#                            self.hidden_size).to(device)
#         c0_2 = torch.zeros(self.num_layers, input2.size(0),
#                            self.hidden_size).to(device)
#
#         out2, _ = self.lstm1(input2, (h0_2, c0_2))
#
#         multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :],out2[:, -1, :]),  -1)
#
#         multi_out0 = self.fc0(multi_out)
#         multi_out1 = self.fc1(multi_out)
#         return multi_out0, multi_out1


class robustlog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(robustlog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)


    def forward(self, features, device):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


#log key add embedding
class loganomaly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim):
        super(loganomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding_dim = embedding_dim
        self.embedding_size = vocab_size
        self.embedding = nn.Embedding(self.embedding_size, self.embedding_dim)
        torch.nn.init.uniform_(self.embedding.weight)
        self.embedding.weight.requires_grad = True

        self.lstm0 = nn.LSTM(self.embedding_dim,
                            hidden_size,
                            num_layers,
                            batch_first=True)

        self.lstm1 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, features, device):
        input0, input1 = features[0], features[1]
        embed0 = self.embedding(input0)

        h0_0 = torch.zeros(self.num_layers, embed0.size(0),
                           self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_layers, embed0.size(0),
                           self.hidden_size).to(device)

        out0, _ = self.lstm0(embed0, (h0_0, c0_0))

        h0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)

        out1, _ = self.lstm1(input1, (h0_1, c0_1))

        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out

