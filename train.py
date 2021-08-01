from nn import NeuralNet
from trainer import Trainer
from optimizer import Optimizer
from config import cfg
from data_utils import get_dataloader
import argparse
import logging
import sys


logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(module)s] %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %I:%M:%S%p", stream=sys.stdout)

logger = logging.getLogger(__file__)


def train(network):
    train_loader, val_loader = get_dataloader(cfg.DATASET.NAME, cfg.DATASET.PATH, cfg.TRAINING.HOLDOUT,
                                              cfg.TRAINING.BATCH_SIZE, cfg.TRAINING.TARGET_SMOOTHING)
    optimizer = Optimizer(network, cfg.TRAINING.LOSS, cfg.TRAINING.LR, cfg.TRAINING.LR_SCHEDULE, cfg.TRAINING.MOMENTUM,
                          cfg.TRAINING.WEIGHT_DECAY)
    train_interface = Trainer(network, optimizer, train_loader, val_loader)

    train_interface.train(cfg.TRAINING.EPOCHS)


def test(network):
    test_loader = get_dataloader(cfg.DATASET.NAME, cfg.DATASET.PATH, 0,
                                 cfg.TRAINING.BATCH_SIZE, cfg.TRAINING.TARGET_SMOOTHING, test=True)

    interface = Trainer(network, None, None, test_loader)
    interface.validate()
    accuracy = sum(interface.val_accuracy) / len(interface.val_accuracy)
    logger.info(f'TEST Accuracy: {accuracy:.3f}')


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
