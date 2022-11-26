import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence


class VAEEncoder(nn.Module):
    def __init__(self, vocab, hidden_size, num_layers, bidirectional, dropout):
        super(VAEEncoder, self).__init__()

        num_embeddings, embedding_dim = len(vocab), vocab.vectors.size(1)
        self.x_emb = nn.Embedding(num_embeddings, embedding_dim, vocab.pad)
        self.x_emb.weight.data.copy_(vocab.vectors)

        self.encoder_rnn = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

    def forward(self, x):
        x = [self.x_emb(i_x) for i_x in x]
        x = pack_sequence(x)

        _, h = self.encoder_rnn(x, None)

        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        return h


class VAEDecoder(nn.Module):
    def __init__(self, x_emb, latent_size, hidden_size, num_layers, dropout):
        super(VAEDecoder, self).__init__()

        self.x_emb = x_emb

        self.decoder_lat = nn.Linear(latent_size, hidden_size)

        self.decoder_rnn = nn.GRU(
            x_emb.embedding_dim + latent_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.decoder_fc = nn.Linear(hidden_size, x_emb.num_embeddings)

    def forward(self, x, z):
        lengths = [len(i_x) for i_x in x]

        x_pad = pad_sequence(x, batch_first=True, padding_value=self.x_emb.padding_idx)
        x_emb = self.x_emb(x_pad)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = pack_padded_sequence(x_input, lengths, batch_first=True)

        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)

        output, _ = pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x_pad[:, 1:].contiguous().view(-1),
            ignore_index=self.x_emb.padding_idx
        )

        return y, recon_loss


class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate"""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


class Vanilla2DDrawingDecoder(nn.Module):
    def __init__(self, latent_dim, enc_out_dim, input_height=200):
        super(Vanilla2DDrawingDecoder, self).__init__()

        modules = []

        self.decoder_input = nn.Linear(latent_dim, enc_out_dim)
        self.num_input_channels = enc_out_dim // 16

        hidden_dims = [self.num_input_channels,
                       self.num_input_channels * 2,
                       self.num_input_channels * 4,
                       self.num_input_channels * 2,
                       self.num_input_channels]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.upscale = Interpolate(size=input_height // 2)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1],
                      out_channels=3,
                      kernel_size=3,
                      padding=1),
            nn.Tanh())

    def forward(self, z):
        result = self.decoder_input(z)

        result = result.view(result.size(0), self.num_input_channels, 4, 4)

        result = self.decoder(result)
        result = self.upscale(result)

        result = self.final_layer(result)
        return result


class Vanilla2DDrawingEncoder(nn.Module):
    def __init__(self, enc_out_dim):
        super(Vanilla2DDrawingEncoder, self).__init__()

        modules = []
        in_channels = 3
        for h_dim in [32, 64, 128, 64, 32]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(32 * 7 * 7, enc_out_dim)

    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)

        hidden = self.fc(features)

        return hidden


class M2DDrawingEncoder(nn.Module):
    def __init__(self, hidden_size, drop_rate):
        super(M2DDrawingEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, 2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, 2)
        self.bn5 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = x.view(-1, 32 * 5 * 5)

        out = self.fc1(x)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out
