import copy
import random

import torch
from torch.utils.data import DataLoader
from torch.distributions.laplace import Laplace

from utils.data import UserItemRatingDataset
from utils.metrics import MetronAtK
from .tools import aggregateByMatrix
from .tools import updateMatrix


class Engine(object):
    """
    Meta Engine for training & evaluating our model

    Note: Subclass should implement self.model!

    """

    def __init__(self, config):
        self.config = config  # model configuration
        self.server_model_params = {}
        self.client_model_params = {}
        self._metron = MetronAtK(top_k=self.config['top_k'])

    def instanceUserTrainLoader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True), len(dataset)

    def federatedTrainSingleBatch(self, model_client, batch_data, optimizer, server_params=None, optimizer_adapter=None):
        """train a batch and return an updated model."""
        from model.loss import Loss

        # load batch data.
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        model_loss = Loss(self.config)

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()
            model_loss = model_loss.cuda()

        if optimizer_adapter:
            # update adapter.
            optimizer_adapter.zero_grad()
            ratings_adapter = model_client(items, server_params)
            loss_adapter = model_loss(ratings_adapter.view(-1), ratings)
            loss_adapter.backward()
            optimizer_adapter.step()
        
        # update embeddings.
        optimizer.zero_grad()
        ratings_pred = model_client(items, server_params)
        loss_pred = model_loss(ratings_pred.view(-1), ratings)
        loss_pred.backward()
        optimizer.step()

        if self.config['use_cuda']:
            loss_pred = loss_pred.cpu()

        return model_client, loss_pred.item()

    def aggregateParamsFromClients(self, client_params, **kwargs):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # KeyError
        if 'aggregation_matrix' not in kwargs or kwargs['aggregation_matrix'] is None:
            raise ValueError("The 'aggregation_matrix' argument cannot be empty.")
        # aggregating
        self.server_model_params = aggregateByMatrix(client_params, kwargs["aggregation_matrix"], self.config)
        pass

    def federatedTrainOneRound(self, train_data, item_embeddings, mlp_weights, iteration):
        """sample users participating in single round."""

        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)

        # store users' model parameters of current round.
        participant_params = {}
        # store all the users' train loss/
        losses = {}
        # store all the clients' sample num
        client_sample_num = {}

        # perform model update for each participated user.
        for user in participants:
            loss = 0

            # for the first round, client models copy initialized parameters directly.
            client_model = copy.deepcopy(self.model)
            client_model.setItemEmbeddings(item_embeddings)

            # for other rounds, client models receive updated item embedding and score function from server.
            if iteration != 0:
                client_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        client_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data)

                if self.config['use_cuda']:
                    for key in client_param_dict.keys():
                        client_param_dict[key] = client_param_dict[key].cuda()

                client_model.load_state_dict(client_param_dict)
                if user in self.server_model_params:
                    client_model.adpatItemEmbeddings(self.server_model_params[user])
                # adapt_weight[user] = client_model.adapt_weight.detach().cpu()

            # Defining optimizers
            if self.config['backbone'] == 'FCF':
                optimizer = torch.optim.Adam([
                    {'params': client_model.user_embedding.parameters(), 'lr': self.config['lr_embedding']},
                    {'params': client_model.item_embeddings.parameters(), 'lr': self.config['lr_embedding']}
                ])
            
            optimizer_adapter = torch.optim.Adam([
                    {'params': client_model.adapter.parameters(), 'lr': self.config['lr_adapter']}
                ])
            
            # load current user's training data and instance a train loader.
            user_train_data = [train_data[0][user], train_data[1][user], train_data[2][user]]
            # record the current user's sample num
            user_dataloader, user_sample_num = self.instanceUserTrainLoader(user_train_data)
            client_sample_num[user] = user_sample_num

            client_model.train()
            sample_num = 0
            # update client model.
            client_losses = []
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    if iteration > 0 and user in self.server_model_params:
                        client_model, client_loss = self.federatedTrainSingleBatch(client_model, batch, optimizer, self.server_model_params[user], optimizer_adapter)
                    else:
                        client_model, client_loss = self.federatedTrainSingleBatch(client_model, batch, optimizer)  #, self.server_model_params[user]
                    loss += client_loss * len(batch[0])
                    sample_num += len(batch[0])

                losses[user] = loss / sample_num
                client_losses.append(loss / sample_num)

                # check convergence, avoid division by zero
                if epoch > 0 and abs(client_losses[epoch] - client_losses[epoch - 1]) / abs(
                        client_losses[epoch - 1] + 1e-5) < self.config['threshold']:
                    break

            # obtain client model parameters,
            self.client_model_params[user] = copy.deepcopy(client_model.state_dict())

            # store client models' local parameters for global update.
            participant_params[user] = copy.deepcopy(self.client_model_params[user])

            # delete all user-related data
            del participant_params[user]['user_embedding.weight']
        
        # aggregate client models in server side.
        agg_num = int(len(participants) * self.config["agg_clients_ratio"])
        agg_participants = random.sample(list(participant_params.keys()), agg_num)
        agg_participants_params = {user: participant_params[user] for user in agg_participants}
        agg_client_sample_num = {user: client_sample_num[user] for user in agg_participants}

        aggregation_matix = updateMatrix(agg_participants_params, agg_client_sample_num, self.config)
        self.aggregateParamsFromClients(agg_participants_params, aggregation_matrix=aggregation_matix)

        return losses

    @torch.no_grad()
    def federatedEvaluate(self, evaluate_data):
        # evaluate all client models' performance using testing data.
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]

        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()

        # store all users' test item prediction score.
        test_scores = None
        # store all users' negative items prediction scores.
        negative_scores = None

        for user in range(self.config['num_users']):
            user_model = copy.deepcopy(self.model)

            if user in self.client_model_params.keys():
                user_param_dict = copy.deepcopy(self.client_model_params[user])
                for key in user_param_dict.keys():
                    user_param_dict[key] = user_param_dict[key].data
            else:
                user_param_dict = copy.deepcopy(self.model.state_dict())

            user_model.load_state_dict(user_param_dict)
            if user in self.server_model_params:
                user_model.adpatItemEmbeddings(self.server_model_params[user])

            user_model.eval()

            # obtain user's positive test information.
            test_item = test_items[user: user + 1]
            # obtain user's negative test information.
            negative_item = negative_items[user * 99: (user + 1) * 99]
            # perform model prediction.
            test_score = user_model(test_item)
            negative_score = user_model(negative_item)

            if user == 0:
                test_scores = test_score
                negative_scores = negative_score
            else:
                test_scores = torch.cat((test_scores, test_score))
                negative_scores = torch.cat((negative_scores, negative_score))

        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]

        hr, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()

        return hr, ndcg
