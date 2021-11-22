import os
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
import numpy as np
from sklearn.metrics import classification_report

from .evaluation import validate
from model.model import CRNN


def train(args, pt_dir, chkpt_path, trainloader, devloader, testloader, writer, logger, hp, hp_str):

    model = CRNN(hp).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.adam)

    epoch = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['step']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")

    try:
        criterion = nn.CrossEntropyLoss()
        while epoch < hp.train.epoch:
            epoch += 1
            model.train()
            loss = 0

            for data, target in tqdm(trainloader):
                data = data.cuda()
                target = target.cuda().long()

                output = model(data)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, epoch))
                    raise Exception("Loss exploded")

            if epoch % hp.train.summary_interval == 0:
                writer.log_training(loss, epoch)
                logger.info("Wrote summary at step %d" % epoch)

            if epoch % hp.train.checkpoint_interval == 0:
                save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % epoch)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': epoch,
                    'hp_str': hp_str,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)
                validate(model, devloader, writer, epoch)
        test(model, testloader)

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()



def test(model, testloader):
    with torch.no_grad():
        model.eval()

        outputs = []
        targets = []

        for data, target in tqdm(testloader):
            data = data.cuda()
            output = model(data)
            output = F.softmax(output, dim=1)

            outputs += output.cpu().detach().numpy().tolist()
            targets += target.tolist()

        print(classification_report(targets, np.argmax(outputs, axis=1)))
