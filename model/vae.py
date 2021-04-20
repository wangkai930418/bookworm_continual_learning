import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, num_labels=0):
        super().__init__()
        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        self.latent_size = latent_size
        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, num_labels)

    def forward(self, x, c):
        if x.dim() > 2:
            x = x.view(-1)
        batch_size = x.size(0)
        means, log_var = self.encoder(x, c)
        std = torch.exp(0.5 * log_var).cuda()
        eps = torch.randn([batch_size, self.latent_size]).cuda()
        z = eps * std + means
        recon_x = self.decoder(z, c)
        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):
        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).cuda()
        recon_x = self.decoder(z, c)
        return recon_x


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, num_labels):
        super().__init__()
        in_size, hid_size, out_size = layer_sizes
        in_size+=num_labels

        self.MLP = nn.Sequential(nn.Linear(in_size, hid_size),
                                 nn.ReLU(),
                                 nn.Dropout(0.7),
                                 nn.Linear(hid_size, out_size),
                                 nn.ReLU(),
                                 )
        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c):
        x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size,  num_labels):
        super().__init__()
        hid_size,out_size=layer_sizes
        input_size = latent_size + num_labels

        self.MLP = nn.Sequential(nn.Linear(input_size, hid_size),
                                 nn.ReLU(),
                                 nn.Dropout(0.5), # make training stable
                                 nn.Linear(hid_size,out_size),
                                 )

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)
        return x

