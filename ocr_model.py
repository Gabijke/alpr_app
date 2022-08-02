import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class BlockRNN(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, bidirectional):
        super(BlockRNN, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.bidirectional = bidirectional
        self.gru = nn.LSTM(in_size, hidden_size, bidirectional=bidirectional, batch_first=True)

    def forward(self, batch: torch.float32, add_output=False):
        outputs, _ = self.gru(batch)
        out_size = int(outputs.size(2) / 2)
        if add_output:
            outputs = outputs[:, :, :out_size] + outputs[:, :, out_size:]
        return outputs


class OcrNet(nn.Module):

    def __init__(self,
                 bidirectional: bool = True,
                 rnn_hidden_size: int = 512,
                 rnn_input_size: int = 32,
                 vocab_size: int = 23  # 12[alpha] + 10[digit] + 1[blank]
                 ):
        super(OcrNet, self).__init__()

        # conv block
        resnet = resnet18(pretrained=True)
        resnet._modules['conv1'] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules_resnet = list(resnet.children())[:-3]
        self.conv = nn.Sequential(*modules_resnet)

        # rnn block
        self.linear1 = nn.Linear(1024, rnn_input_size)
        self.gru1 = BlockRNN(rnn_input_size, rnn_hidden_size, rnn_hidden_size, bidirectional=bidirectional)
        self.gru2 = BlockRNN(rnn_hidden_size, rnn_hidden_size, vocab_size, bidirectional=bidirectional)
        self.linear2 = nn.Linear(rnn_hidden_size * 2, vocab_size)

    def forward(self, batch: torch.float32):
        batch_size = batch.size(0)

        # convolutions
        batch = self.conv(batch)

        # make sequences
        batch = batch.permute(0, 3, 1, 2)
        n_channels = batch.size(1)
        batch = batch.reshape(batch_size, n_channels, -1)
        batch = self.linear1(batch)

        # rnn layers
        batch = self.gru1(batch, add_output=True)
        batch = self.gru2(batch)

        batch = self.linear2(batch)
        batch = batch.permute(1, 0, 2)
        batch = F.log_softmax(batch, dim=2)
        return batch
