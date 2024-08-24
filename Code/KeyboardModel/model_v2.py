import torch
import torch.nn as nn
import torch.nn.functional as F


class TypingLSTMV2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_type, output_size=1, dropout=0.3):
        super(TypingLSTMV2, self).__init__()

        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        if model_type not in ['many', 'one']:
            raise ValueError("model_type must be either 'many' or 'one'")

        self.model_type = model_type

    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Apply layer normalization
        out = self.layer_norm(out)

        # Apply the fully connected layer to each time step
        out = self.fc(out)

        return out  # output of shape (batch_size, seq_length, output_size)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, pos_weight=None):
        """
        :param alpha: alpha is the weight of the positive class. If it is higher than 0.5, the model will be more focused on the positive class.
        :param gamma: Gamma is the focusing parameter. If it is higher than 1, the model will be more focused on the hard examples.
        :param pos_weight:
        """

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # Calculate the probabilities
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Calculate the cross entropy loss, which is log(p_t)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction="none")

        # Calculate the focal loss
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()  # Average the loss over the batch
