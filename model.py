from __future__ import print_function
import numpy as np
from sklearn.preprocessing import normalize

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from progressbar import ProgressBar
import time

beijing_station = ["dongsi", "tiantan", "guanyuan", "wanshouxigong", "aotizhongxin", \
                    "nongzhanguan", "wanliu", "beibuxinqu", "zhiwuyuan", "fengtaihuayuan", \
                    "yungang", "gucheng", "fangshan", "daxing", "yizhuang", \
                    "tongzhou", "shunyi", "pingchang", "mentougou", "pinggu", \
                    "huairou", "miyun", "yanqin", "dingling", "badaling", \
                    "miyunshuiku", "donggaocun", "yongledian", "yufa", "liulihe", \
                    "qianmen", "yongdingmennei", "xizhimenbei", "nansanhuan", "dongsihuan"]

london_station = ['CD1', 'BL0', 'GR4', 'MY7', 'HV1', 'GN3', 'GR9', 'LW2', 'GN0', 'KF1', 'CD9', 'ST5', 'TH4']

try:
    import tensorflow as tf
except ImportError:
    print('! TensorFlow not installed. No tensorflow logging.')
    tf = None


def tf_log(writer, key, value, epoch):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, epoch)


def make_train_valid_dataset(name, length=24, train_percentage=0.7):

    print('> reading %s ...' % name)
    aq_matrix = np.load(open(name, 'rb'))
    total_count, dim = aq_matrix.shape
    print('> get matrix of shape', total_count, dim)

    max_values = np.max(aq_matrix, axis=0)
    aq_matrix = aq_matrix / max_values

    train_matrix = aq_matrix[:int(train_percentage * total_count), :]
    valid_matrix = aq_matrix[int(train_percentage * total_count):, :]

    train_dataset = AQDataset(train_matrix, length)
    valid_dataset = AQDataset(valid_matrix, length)
    return train_dataset, valid_dataset, max_values

def smape(predicted, actual):
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)

    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0, casting='unsafe'))


class AQDataset(Dataset):
    def __init__(self, aq_matrix, length):
        super(AQDataset, self).__init__()
        # (total_count, dim)

        self.total_count, self.dim = aq_matrix.shape
        self.length = length
        self.matrix = aq_matrix

    def __getitem__(self, index):
        input_matrix = self.matrix[index: index + self.length, :]
        target = self.matrix[index + 1: index + 1 + self.length, :]

        input_matrix = torch.from_numpy(input_matrix).float()
        target = torch.from_numpy(target).float()

        return input_matrix, target
        # [length, dim] [length, dim]

    def __len__(self):
        return self.total_count - self.length


class BaseModel(nn.Module):
    def __init__(self, dim, length, hid_dim):
        super(BaseModel, self).__init__()

        self.rnn = nn.LSTM(hid_dim, hid_dim, 1, batch_first=True)

        self.emb = nn.Linear(dim, hid_dim)
        self.output = nn.Linear(hid_dim, dim)
        self.dim = dim
        self.length = length
        self.hid_dim = hid_dim

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = (1, batch, self.hid_dim)
        return (Variable(weight.new(*hid_shape).zero_()),
                Variable(weight.new(*hid_shape).zero_()))

    def forward(self, x):
        # x [batch, length/2, dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        x = self.emb(x)
        output, hidden = self.rnn(x, hidden)
        output = self.output(output)

        return output

def train(station, type):
    if type == 0:
        feature_dim = 6
    else:
        feature_dim = 3
    use_length = 24
    rnn_hid_dim = 64
    lr = 0.001
    batch_size = 256
    num_epochs = 40
    grad_clip_rate = 0.5
    epoch_per_display = 4
    predict_time = 48

    output = './output/tf_log'
    if type == 0:
        train_dataset, valid_dataset, max_values = make_train_valid_dataset('./files/bj_aq_'+station+'.npy', use_length)
    else:
        train_dataset, valid_dataset, max_values = make_train_valid_dataset('./files/ld_aq_'+station+'.npy', use_length)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=1)
    train_writer = tf and tf.summary.FileWriter(os.path.join(output, 'train/'+station+'/'))
    valid_writer = tf and tf.summary.FileWriter(os.path.join(output, 'valid/'+station+'/'))

    model = BaseModel(feature_dim, use_length, rnn_hid_dim)
    #model = model.cuda()
    #model = nn.DataParallel(model).cuda()
    crit = nn.MSELoss(reduce=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in xrange(num_epochs):

        total_loss = 0.

        for i, (x, t) in enumerate(train_loader):
            #x = Variable(x, requires_grad=True).cuda()
            #t = Variable(t, requires_grad=False).cuda()
            x = Variable(x, requires_grad=True)
            t = Variable(t, requires_grad=False)
            optimizer.zero_grad()
            output = model(x)
            loss = crit(output, t)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), grad_clip_rate)
            optimizer.step()

            total_loss += loss.item() * x.size(0)


        total_loss /= len(train_loader)

        if epoch % epoch_per_display == 0:

            # evaluate
            valid_loss = 0.
            valid_accuracy = 0.
            for i, (x, t) in enumerate(valid_loader):
                #x = Variable(x, volatile=True).cuda()
                #t = Variable(t, volatile=True).cuda()
                x = Variable(x, volatile=True)
                t = Variable(t, volatile=True)
                output = model(x)
                loss = crit(output, t)
                valid_loss += loss.item() * x.size(0)
            valid_loss /= len(valid_loader)

            print('> epoch {}, train loss {}, valid loss {}'.format(epoch, total_loss, valid_loss))
            #print('> epoch {}, train loss {}, valid loss {}, valid accuracy {}'.format(epoch, total_loss, valid_loss, valid_accuracy))

            # write to tf log
            tf_log(train_writer, 'loss', total_loss, epoch)
            tf_log(valid_writer, 'loss', valid_loss, epoch)
            train_writer.flush()
            valid_writer.flush()

    torch.save(model.state_dict(), './files/params_'+station+'.pkl')



if __name__ == '__main__':
    for station in london_station:
        train(station, 1)
    for station in beijing_station:
        train(station, 0)
