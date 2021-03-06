import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Accept hidden layers as an argument <num_layer x batch_size x hidden_size> for each hidden and cell state.
    At every forward call, output probability vector of <batch_size x output_size>.
    input_size: N_LETTER
    hidden_size: Size of the hidden dimension
    output_size: N_LETTER
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, padding_idx: int, embed_size: int = 8,
                 num_layers: int = 4, drop_out: float = 0.2):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embed_size = embed_size

        # Initialize LSTM - Notice it does not accept output_size
        self.len_embed = nn.Embedding(hidden_size, embed_size)
        self.embed = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size * 2, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop_out)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input: torch.Tensor, lng_input: torch.Tensor, hidden: torch.Tensor):
        """
        Run LSTM through 1 time step
        SHAPE REQUIREMENT
        - input: <1 x batch_size x N_LETTER>
        - hidden: (<num_layer x batch_size x hidden_size>, <num_layer x batch_size x hidden_size>)
        - lstm_out: <1 x batch_size x N_LETTER>
        """
        input = self.embed(input)
        lng_input = self.len_embed(lng_input)
        input = torch.cat((input, lng_input), dim=1)
        lstm_out, hidden = self.lstm(input.unsqueeze(0), hidden)
        lstm_out = self.fc1(lstm_out)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.softmax(lstm_out)

        return lstm_out, hidden

    def initHidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
