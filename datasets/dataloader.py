import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader

from utils.audio import Audio


def create_dataloader(hp, setname):
    dataset = AMIDataset(hp, setname)

    if setname == 'train':

        num_samples = dataset.num_0 + dataset.num_1
        class_weights = [1.0 / dataset.num_0, 1.0 / dataset.num_1]
        weights = np.array([class_weights[int(label)] for label in dataset.data.labels])
        weights = torch.from_numpy(weights).double()
        weighted_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples)

        return DataLoader(dataset=dataset,
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          sampler=weighted_sampler
                          )
    else:
        return DataLoader(dataset=dataset,
                          batch_size=hp.train.batch_size,
                          shuffle=False, 
                          num_workers=0
                          )


class AMIDataset(Dataset):
    def __init__(self, hp, dataset):
        self.hp = hp
        self.audio = Audio(hp)
        self.seg_len = hp.data.audio_len * hp.audio.sr
        self.num_0 = 1
        self.num_1 = 1

        if dataset == 'train':
            self.filename = hp.set.train
        elif dataset == 'dev':
            self.filename = hp.set.dev
        else:
            self.filename = hp.set.test

        self.data = self.load_data()


    def load_data(self):
        uris = []
        with open(self.filename, 'r') as fin:
            for line in fin.readlines():
                uris.append(line.split('\n')[0])

        df = {'labels': [], 'features': []}
        for uri in tqdm(uris):
            wav_path = self.hp.data.data_dir + uri + '/audio/' + uri + self.hp.data.wav
            wav, _ = librosa.load(wav_path, sr=self.hp.audio.sr)

            with open(self.hp.data.label_dir + uri + '.rttm', 'r') as fin:
                for line in fin.readlines():
                    start = int(float(line.split(' ')[3]) * self.hp.audio.sr)
                    end = int(start + self.seg_len)
                    #df['uri'].append(uri)
                    #df['start'].append(float(line.split(' ')[3]))
                    df['labels'].append(float(line.split(' ')[7]))
                    df['features'].append(self.audio.mfcc(wav[start:end]))
        df = pd.DataFrame(df)

        self.num_0 = len(df.loc[df['labels'] == 0])
        self.num_1 = len(df.loc[df['labels'] == 1])

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        feature = item.features
        label = item.labels
        return feature, label
