import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class NeRFmodel(nn.Module):
    def __init__(
        self,
        embed_pos_L=10,
        embed_direction_L=4,
        hidden_dim_1=256,
        hidden_dim_2=128,
        flag_encoding=True,
    ):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################
        self.embed_pos_L = embed_pos_L
        self.embed_direction_L = embed_direction_L
        self.flag_encoding = flag_encoding

        if self.flag_encoding:
            pos_len = 3 + self.embed_pos_L * 3 * 2
            dir_len = 3 + self.embed_direction_L * 3 * 2
            print("With Encoding")
        else:
            pos_len = 3
            dir_len = 3
            print("No Encoding")

        seed = 1000
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.input_layer = nn.Linear(pos_len, hidden_dim_1)
        self.block_1 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
        )
        self.skip_layer = nn.Linear(hidden_dim_1 + pos_len, hidden_dim_1)
        self.block_2 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_1),
        )
        self.sigma_layer = nn.Linear(hidden_dim_1, hidden_dim_1 + 1)
        self.dir_layer = nn.Linear(hidden_dim_1 + dir_len, hidden_dim_2)
        self.output_layer = nn.Linear(hidden_dim_2, 3)
        self.relu = nn.functional.relu
        self.sigmoid = nn.functional.sigmoid
        self.double()
        self.apply(init_weights)

    def position_encoding(self, x, L, flag):
        #############################
        # Implement position encoding here
        #############################
        if flag:
            y = [x]
            for i in range(L):
                y.append(torch.sin(x * 2**i))
                y.append(torch.cos(x * 2**i))
        else:
            y = [x]

        return torch.cat(y, dim=1)

    def forward(self, pos, direction):
        #############################
        # network structure
        #############################
        encoded_pos = self.position_encoding(pos, self.embed_pos_L, self.flag_encoding)
        encoded_direction = self.position_encoding(
            direction, self.embed_direction_L, self.flag_encoding
        )

        x = self.input_layer(encoded_pos)
        x = self.relu(x)
        x = self.block_1(x)
        x = self.skip_layer(torch.cat((encoded_pos, x), -1))
        x = self.relu(x)
        x = self.block_2(x)
        x = self.sigma_layer(x)
        sigma = x[:, 0]
        hidden_features = x[:, 1:]
        sigma = self.relu(sigma)
        x = self.dir_layer(torch.cat((hidden_features, encoded_direction), -1))
        x = self.relu(x)
        x = self.output_layer(x)
        rgb = self.sigmoid(x)
        return sigma, rgb
