import pathlib
import shutil
import pickle

import torch
import numpy


def train_model(directory):
    l = []
    for i, path in enumerate(pathlib.Path(directory).glob('*')):
        l.append(str(path.resolve()))
        if i % 100 == 0:
            save_model(l, 'trained_model.ckpt')
    save_model(l, 'trained_model.ckpt')


def sample(l):
    i = numpy.random.randint(len(l))
    return l[i]


def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    train_model('./dataset')
    model = load_model('trained_model.ckpt')
    for i in range(10):
        generated = sample(model)
        shutil.copy(
            generated,
            'result/{}{}'.format(i, pathlib.Path(generated).suffix))
