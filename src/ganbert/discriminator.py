import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.LeakyReLU(0.2, inplace=True),
                           nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers)  # per il flatten
        self.logit = nn.Linear(hidden_sizes[-1],
                               num_labels + 1)  # +1 for the probability of this sample being fake/real.
        # self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.sigmoid(logits)
        return last_rep, logits, probs
