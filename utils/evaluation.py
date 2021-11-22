import torch
import torch.nn as nn


def validate(model, devloader, writer, step):
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in devloader:
            data = data.cuda()
            target = target.cuda().long()
            
            output = model(data)
            test_loss = criterion(output, target).item()

            writer.log_evaluation(test_loss, step)
            break

    model.train()
