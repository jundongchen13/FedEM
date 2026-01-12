import copy
import os
import random

import cvxpy as cp
import numpy as np
import torch
import numpy.linalg as la
import matplotlib.pyplot as plt
import torch.nn.functional as F


def aggregateByMatrix(client_params, aggregation_matrix, config):
    client_item_embeddings = torch.zeros(len(client_params), config['num_items'] * config['latent_dim'])
    for i, user in enumerate(client_params.keys()):
        client_item_embeddings[i] = client_params[user]['item_embeddings.weight'].data.view(-1)

    client_item_embeddings = client_item_embeddings.cuda()
    aggregation_matrix = aggregation_matrix.cuda()
    aggregated_item_embedding = aggregation_matrix @ client_item_embeddings

    for i, user in enumerate(client_params.keys()):
        client_params[user]['item_embeddings.weight'].data = aggregated_item_embedding[i].view(config['num_items'], config['latent_dim'])

    return client_params


def updateMatrix(client_params, client_sample_num, config):
    if config['agg_mode'] == 'avg':
        weight_vector = calculate_client_weights(client_sample_num)
        aggregation_matix = weight_vector.unsqueeze(0).repeat(len(weight_vector), 1)
    elif config['agg_mode'] == 'sim':
        aggregation_matix = calculate_similarity(client_params, config)
    elif config['agg_mode'] == 'both':
        aggregation_matix = optimize_similarity(client_params, client_sample_num, config)
    
    aggregation_matix = matrix_shift(aggregation_matix)
    return aggregation_matix

        
def optimize_similarity(client_params, client_sample_num, config):
    n = len(client_params)
    similarity_matrix = - calculate_similarity(client_params, config).cpu().numpy()
    weight_vector = calculate_client_weights(client_sample_num).cpu().numpy()

    optimized_matrix = torch.zeros(n, n)
    p = weight_vector
    P = np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(n):
        s = similarity_matrix[i]
        x = cp.Variable(n)

        objective = cp.Minimize(cp.quad_form(x, P) + (config['alpha'] * 2 * s - 2 * p).T @ x)
        constraints = [G @ x <= h, A @ x == b]
        prob = cp.Problem(objective, constraints)

        prob.solve()
        optimized_matrix[i] = torch.Tensor(x.value)

    return optimized_matrix


def calculate_similarity(client_params, config):
    X = torch.zeros(len(client_params), config['num_items'])
    for i, user in enumerate(client_params.keys()):
        X[i] = PCA(client_params[user]['item_embeddings.weight'].data).flatten()

    X = X.cuda()
    if config['sim_mode'] == 'l2':
        x_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
        dist_sq = x_norm_sq + x_norm_sq.T - 2 * X @ X.T
        dist_sq = torch.clamp(dist_sq, min=0)
        dist = torch.sqrt(dist_sq)
        similarity_matrix = 1 / (dist + 1)

    elif config['sim_mode'] == 'cos':
        X_norm = F.normalize(X, p=2, dim=1)
        similarity_matrix = X_norm @ X_norm.T
        similarity_matrix = similarity_matrix.clamp(min=0)

    similarity_matrix[similarity_matrix > 0.9] = 1
    return similarity_matrix


def calculate_client_weights(client_sample_num):
    if not client_sample_num:
        raise ValueError("The 'client_sample_num' dictionary cannot be empty.")
    total_samples = sum(client_sample_num.values())
    if total_samples == 0:
        raise ValueError("Total sample number must be greater than zero to calculate proportions.")
    
    proportions = [samples / total_samples for samples in client_sample_num.values()]
    return torch.tensor(proportions, dtype=torch.float32)


def matrix_shift(matrix):
    matrix[matrix.abs() < 1e-20] = 0
    row_sums = matrix.sum(dim=1)
    matrix = matrix / row_sums.view(-1, 1)
    return matrix


def PCA(x: torch.Tensor, n_components=1):
    x = x.cuda()
    x = x - x.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    return torch.matmul(x, Vh[:n_components].T)