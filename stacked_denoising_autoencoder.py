# coding: utf-8
import chainer
import chainer.links as L
import chainer.functions as F


class DenoisingAutoEncoder(chainer.Chain):
    def __init__(self, input_size, output_size, encoder_activation=True, decoder_activation=True):
        w = chainer.initializers.Normal(scale=0.01)
        super(DenoisingAutoEncoder, self).__init__()
        with self.init_scope():
            self.encoder = L.Linear(input_size, output_size, initialW=w)
            self.decoder = L.Linear(output_size, input_size, initialW=w)
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation


    def __call__(self, x):
        if self.encoder_activation:
            h = F.relu(self.encoder(F.dropout(x, 0.2)))
        else:
            h = self.encoder(F.dropout(x, 0.2))

        if self.decoder_activation:
            h = F.relu(self.decoder(F.dropout(h, 0.2)))
        else:
            h = self.decoder(F.dropout(h, 0.2))
        return h


    def encode(self, x):
        if self.encoder_activation:
            h = F.relu(self.encoder(x))
        else:
            h = self.encoder(x)
        return h


    def decode(self, x):
        if self.decoder_activation:
            h = F.relu(self.decoder(x))
        else:
            h = self.decoder(x)
        return h


class StackedDenoisingAutoEncoder(chainer.ChainList):
    def __init__(self, input_dim):
        super(StackedDenoisingAutoEncoder, self).__init__(
            DenoisingAutoEncoder(input_dim, 500, decoder_activation=False),
            DenoisingAutoEncoder(500, 500),
            DenoisingAutoEncoder(500, 2000),
            DenoisingAutoEncoder(2000, 10, encoder_activation=False)
        )

    def __call__(self, x):
        # encode
        models = []
        for dae in self.children():
            x = dae.encode(x)
            models.append(dae)

        # decode
        for dae in reversed(models):
            x = dae.decode(x)
        return x