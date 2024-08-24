import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    math:
        {PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        {PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        where pos is the word position and i is the embed idx
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=512).
    Examples:
        pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :].to(x.device)
        return self.dropout(x)


def _generate_square_subsequent_mask(sz):
    return torch.log(torch.tril(torch.ones(sz, sz)))


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, num_tokens, embed_dim, num_heads, num_layers, num_hidden=2048, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.src_mask = None

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.input_emb = nn.Embedding(num_tokens, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=num_hidden, dropout=dropout)
        encoder_norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)

        self.decoder = nn.Linear(embed_dim, num_tokens)

        self.init_weights()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, has_mask=True):
        """
        Args:
            src: the sequence to the encoder (required). Shape of [sequence length, batch size, num token].
            has_mask: whether to use mask or not (default=True).

        Returns:
            output: the output of the decoder. Shape of [sequence length, batch size, num tokens].
        """
        if has_mask:
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                device = src.device
                self.src_mask = _generate_square_subsequent_mask(len(src)).to(device)

        else:
            self.src_mask = None

        # Embedding
        src = self.input_emb(src) * math.sqrt(self.embed_dim)
        # Position
        src = self.pos_encoder(src)

        # Encoder
        output = self.encoder(src, mask=self.src_mask)
        # Decoder
        output = self.decoder(output)

        return output
