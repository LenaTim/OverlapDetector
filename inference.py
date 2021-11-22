import os
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import librosa
import argparse
import numpy as np
import torch.nn.functional as F

from utils.hparams import HParam
from model.model import CRNN
from utils.audio import Audio


def main(args, hp):
    with torch.no_grad():
        audio = Audio(hp)

        uris = []
        with open(hp.set.all, 'r') as fin:
            for line in fin.readlines():
                uris.append(line.split('\n')[0])

        model = CRNN(hp).cuda()
        chkpt_model = torch.load(args.checkpoint_path)['model']
        model.load_state_dict(chkpt_model)
        model.eval()

        for uri in tqdm(uris):
            wav_path = hp.data.data_dir + uri + '/audio/' + uri + hp.data.wav
            wav, _ = librosa.load(wav_path, sr=hp.audio.sr)
            outputs = []
            seg_len = hp.data.audio_len * hp.audio.sr
            step = 0

            while step < len(wav) - seg_len:
                start = int(step)
                end = int(start + seg_len)
                data = audio.stft(wav[start:end])

                data = torch.from_numpy(data)
                data = data.unsqueeze(0).cuda()

                output = model(data)
                output = F.softmax(output, dim=1)

                outputs += output.cpu().detach().numpy().tolist()

                step += seg_len

            outputs = np.argmax(outputs, axis=1)
            filename = hp.data.output_dir + uri + '.rttm'
            with open(filename, 'w') as fout:
                for j, o in enumerate(outputs):
                    line = 'OVERLAP ' + uri + ' 1 ' + str(j*hp.data.audio_len) + ' ' + str(hp.data.audio_len) + ' <NA> <NA> ' + str(o) + ' <NA> <NA>\n'
                    fout.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    
    args = parser.parse_args()

    hp = HParam(args.config)

    main(args, hp)
