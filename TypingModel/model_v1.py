import torch.nn as nn


class TypingLSTMV1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_type, output_size=1):
        super(TypingLSTMV1, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        if model_type not in ['many', 'one']:
            raise ValueError("model_type must be either 'many' or 'one'")

        self.model_type = model_type

    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        out = self.fc(out)  # Apply the fully connected layer to each time step,

        return out  # output of shape (batch_size, seq_length, output_size)
