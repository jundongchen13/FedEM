import copy

import torch
import torch.nn.functional as F

from model.engine import Engine


class Adapter(torch.nn.Module):
    def __init__(self, config):
        super(Adapter, self).__init__()
        # config
        layer_sizes = config['adapter']

        # mlp for adapter
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        layers.append(torch.nn.Sigmoid())
        
        self.mlp = torch.nn.Sequential(*layers)
        # self.param_init()

    def forward(self, local_params, server_params):
        delta_params = server_params - local_params
        inputs = torch.cat((local_params, delta_params), dim=-1)
        weights = self.mlp(inputs)
        # update embeddings
        self.adapt_weight = weights[:,0].unsqueeze(1)
        weighted_params = local_params + self.adapt_weight * delta_params
        return weighted_params


class FCF(torch.nn.Module):
    def __init__(self, config):
        super(FCF, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.adapter = Adapter(self.config)
        self.item_embeddings = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.user_embedding = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)

        self.logistic = torch.nn.Sigmoid()
        # self._init_weight_()

    def setItemEmbeddings(self, item_embeddings):
        self.item_embeddings = copy.deepcopy(item_embeddings)

    def adpatItemEmbeddings(self, server_model_param):
        local_item_embeddings = self.item_embeddings.weight.data
        global_item_embeddings = server_model_param['item_embeddings.weight'].data
        self.item_embeddings.weight.data = self.adapter(local_item_embeddings, global_item_embeddings)
    
    def forward(self, item_indices, server_model_param=None):
        if server_model_param:
            local_item_embeddings = self.item_embeddings(item_indices)
            global_item_embeddings = server_model_param['item_embeddings.weight'].data[item_indices,:]
            item_embeddings = self.adapter(local_item_embeddings, global_item_embeddings)
        else:   
            item_embeddings = self.item_embeddings(item_indices)

        user_embedding = self.user_embedding.weight
        rating = torch.mul(user_embedding, item_embeddings)
        rating = torch.sum(rating, dim=1)
        rating = self.logistic(rating)

        return rating


class ModelEngine(Engine):
    """Engine for training & evaluating FCF model"""

    def __init__(self, config):
        model_name = config['backbone']
        self.model = eval(model_name)(config)
        if config['use_cuda'] is True:
            self.model.cuda()
        super(ModelEngine, self).__init__(config)
        print(self.model)
