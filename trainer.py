import os
import time
import logging
import argparse

from utils.train import train
from utils.hparams import HParam
from utils.writer import MyWriter
from datasets.dataloader import create_dataloader

import warnings
warnings.simplefilter("ignore", UserWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="config.yaml file")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Name of the model. Used for both logging and saving checkpoints.")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    pt_dir = os.path.join(hp.log.chkpt_dir, args.model)
    os.makedirs(pt_dir, exist_ok=True)

    log_dir = os.path.join(hp.log.log_dir, args.model)
    os.makedirs(log_dir, exist_ok=True)

    chkpt_path = args.checkpoint_path if args.checkpoint_path is not None else None

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (args.model, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    writer = MyWriter(hp, log_dir)

    trainloader = create_dataloader(hp, 'train')
    devloader = create_dataloader(hp, 'dev')
    testloader = create_dataloader(hp, 'test')

    print('train len: ', len(trainloader))
    print('dev len: ', len(devloader))
    print('test len: ', len(testloader))

    train(args, pt_dir, chkpt_path, trainloader, devloader, testloader, writer, logger, hp, hp_str)
