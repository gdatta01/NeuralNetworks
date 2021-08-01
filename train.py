from nn import NeuralNet
from trainer import Trainer
from dataloader import Dataloader
from config import cfg
from utils import read_data, read_labels, convert_labels
import os
import argparse
import logging
import numpy as np
import sys


logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(module)s] %(asctime)s: %(message)s", datefmt="%Y-%m-%d %I:%M:%S%p", stream=sys.stdout)

logger = logging.getLogger(__file__)

def train(network):
    train_data_path = os.path.join(cfg.DATASET.LOCATION, cfg.DATASET.TRAIN.DATA)
    train_labels_path = os.path.join(cfg.DATASET.LOCATION, cfg.DATASET.TRAIN.LABELS)

    data = read_data(train_data_path, cfg.DATASET.TRAIN.SIZE, network.inputs)
    labels = convert_labels(read_labels(train_labels_path, cfg.DATASET.TRAIN.SIZE), network.outputs, cfg.TRAINING.TARGET_SMOOTHING)

    loader = Dataloader(data, labels, cfg.TRAINING.BATCH_SIZE)

    train_interface = Trainer(network, cfg.TRAINING.LOSS, loader, cfg.TRAINING.LR, cfg.TRAINING.LR_SCHEDULE, cfg.TRAINING.MOMENTUM)

    train_interface.train(cfg.TRAINING.EPOCHS)


def test(network):
    test_data_path = os.path.join(cfg.DATASET.LOCATION, cfg.DATASET.TEST.DATA)
    test_labels_path = os.path.join(cfg.DATASET.LOCATION, cfg.DATASET.TEST.LABELS)

    test_data = read_data(test_data_path, cfg.DATASET.TEST.SIZE, network.inputs)
    test_labels = read_labels(test_labels_path, cfg.DATASET.TEST.SIZE)
    test_loader = Dataloader(test_data, test_labels, cfg.DATASET.TEST.SIZE)

    outputs = network(next(test_loader())[0])
    accuracy = np.count_nonzero(np.argmax(outputs, axis=1) == test_labels) / cfg.DATASET.TEST.SIZE
    logger.info(f'TEST Accuracy: {accuracy}')
    


def get_args():
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('--cfg', type=str, help='Config containing network, dataset, and training information')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    cfg.merge_from_file(args.cfg)

    net_cfg = cfg.NETWORK
    nn = NeuralNet(net_cfg.INPUTS, net_cfg.HIDDEN_LAYERS, net_cfg.OUTPUTS)

    train(nn)