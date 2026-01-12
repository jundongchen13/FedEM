import argparse
import datetime
import logging
import os
import time

import torch

from utils.data import SampleGenerator
from utils.utils import setSeed, initLogging, loadData, format_arg_str
# import pandas as pd


def loadEngine(configuration):
    # Load engine according to the alias
    from model.model import ModelEngine
    load_engine = ModelEngine(configuration)

    return load_engine


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='FCF', choices=['FCF'])
    parser.add_argument('--dataset', type=str, default='lastfm-2k')
    parser.add_argument('--data_file', type=str, default='ratings.dat')
    parser.add_argument('--train_frac', type=float, default=1.0)
    parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--global_round', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr_embedding', type=float, default=1e-1)
    parser.add_argument('--lr_adapter', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--num_negative', type=int, default=4)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--agg_clients_ratio', type=float, default=0.1)
    parser.add_argument('--agg_mode', type=str, default='both', choices=['avg', 'sim', 'both'])
    parser.add_argument('--sim_mode', type=str, default='l2', choices=['l2', 'cos'])

    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--adapter', type=list, default=[32, 16, 8, 1])

    args = parser.parse_args()

    # Config
    config = vars(args)

    # Set cuda
    if config['use_cuda'] is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_id'])

    # Set random seed
    setSeed(config['seed'])

    # Logging.
    path = 'logs/'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    log_file_name = os.path.join(path,
                                 '[{}]-[{}.{}]-[{}].txt'.format(config['backbone'], config['dataset'],
                                                                config['data_file'].split('.')[0],
                                                                current_time))
    initLogging(log_file_name)

    # Load Data
    ratings, config['num_users'], config['num_items'] = loadData('./datasets', config['dataset'], config,
                                                                 config['data_file'])

    # ratings.to_csv('filmtrust.csv', index=False)
    engine = loadEngine(config)

    args_prient = format_arg_str(config, exclude_lst=[])

    logging.info(args_prient)

    # DataLoader for training
    sample_generator = SampleGenerator(ratings=ratings, config=config)
    validate_data = sample_generator.validate_data
    test_data = sample_generator.test_data

    # Initialize for training
    test_hrs, test_ndcgs, val_hrs, val_ndcgs, train_losses = [], [], [], [], []
    best_test_hr, final_test_round = 0, 0

    item_embeddings_init = torch.nn.Embedding(num_embeddings=config['num_items'], embedding_dim=config['latent_dim'])

    mlp_weights_init = None

    if config['use_cuda']:
        item_embeddings_init = item_embeddings_init.cuda()

    times = []

    for iteration in range(config['global_round']):

        logging.info('--------------- Round {} starts ! ---------------'.format(iteration + 1))

        if config['backbone'] == 'FCF':
            train_data = sample_generator.store_all_train_data(config['num_negative'])
        else:
            train_data = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])

        # 1. Train Phase
        start_time = time.perf_counter()
        train_loss = engine.federatedTrainOneRound(train_data, item_embeddings_init, mlp_weights_init, iteration)
        end_time = time.perf_counter()

        times.append((end_time - start_time))

        logging.info('[{}/{}][{}] Time consuming: {:.4f}'.format(config['dataset'],
                                                                 config['data_file'],
                                                                 config['backbone'],
                                                                 (end_time - start_time)))

        loss = sum(train_loss.values()) / len(train_loss.keys())
        train_losses.append(loss)

        logging.info(
            '[Epoch {}/{}][Train] Loss = {:.4f}'.format(iteration + 1, config['global_round'], loss))

        # 2. Evaluations on Validation set
        val_hr, val_ndcg = engine.federatedEvaluate(validate_data)

        logging.info(
            '[Epoch {}/{}][Validation] HR@{} = {:.4f}, NDCG@{} = {:.4f}'.format(iteration + 1, config['global_round'],
                                                                                config['top_k'], val_hr,
                                                                                config['top_k'],
                                                                                val_ndcg))

        val_hrs.append(val_hr)
        val_ndcgs.append(val_ndcg)

        # 3. Evaluations on Test set
        hr, ndcg = engine.federatedEvaluate(test_data)

        logging.info(
            '[Epoch {}/{}][Test] HR@{} = {:.4f}, NDCG@{} = {:.4f}'.format(iteration + 1, config['global_round'],
                                                                          config['top_k'], hr, config['top_k'], ndcg))

        test_hrs.append(hr)
        test_ndcgs.append(ndcg)

        # Choose the model has the best performances
        if hr >= best_test_hr:
            best_test_hr = hr
            final_test_round = iteration

    logging.info('--------------- The model training is finished ---------------')

    logging.info('[{}/{}][{}] Time consuming: {:.4f}'.format(config['dataset'],
                                                             config['data_file'],
                                                             config['backbone'],
                                                             sum(times)))

    # use a dict format to save results
    content = config.copy()

    # delete some unuseful key-value
    del content['device_id']
    del content['use_cuda']

    logging.info(str(content))

    # add some useful key-value
    content['finish_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    content['hr'] = val_hrs[final_test_round]
    content['ndcg'] = val_ndcgs[final_test_round]

    logging.info('--------------- Metrics For Top {} ---------------'.format(config['top_k']))
    logging.info('loss_list: {}'.format(train_losses))
    logging.info('hit_list: {}'.format(test_hrs))
    logging.info('ndcg_list: {}'.format(test_ndcgs))

    notice = 'Best test hr: {:.4f}, ndcg: {:.4f} at round {}'.format(test_hrs[final_test_round],
                                                                     test_ndcgs[final_test_round], final_test_round + 1)

    logging.info(notice)
