import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import BaseModel, AQDataset
from model import make_train_valid_dataset
from model import smape
import os, json, sys

use_length = 24
rnn_hid_dim = 64
predict_time = 48

beijing_output = []
london_output = []

beijing_station = ["dongsi", "tiantan", "guanyuan", "wanshouxigong", "aotizhongxin", \
                    "nongzhanguan", "wanliu", "beibuxinqu", "zhiwuyuan", "fengtaihuayuan", \
                    "yungang", "gucheng", "fangshan", "daxing", "yizhuang", \
                    "tongzhou", "shunyi", "pingchang", "mentougou", "pinggu", \
                    "huairou", "miyun", "yanqin", "dingling", "badaling", \
                    "miyunshuiku", "donggaocun", "yongledian", "yufa", "liulihe", \
                    "qianmen", "yongdingmennei", "xizhimenbei", "nansanhuan", "dongsihuan"]

london_station = ['CD1', 'BL0', 'GR4', 'MY7', 'HV1', 'GN3', 'GR9', 'LW2', 'GN0', 'KF1', 'CD9', 'ST5', 'TH4']


def demo_test():
    feature_dim = 6
    model = BaseModel(feature_dim, use_length, rnn_hid_dim)
    model.load_state_dict(torch.load('params.pkl'))
    train_dataset, valid_dataset, max_values = make_train_valid_dataset('bj_aq.npy', use_length)
    # sample
    x, t = valid_dataset[0]
    #x = Variable(x, volatile=True).cuda()
    #t = Variable(t, volatile=True).cuda()
    x = Variable(x, volatile=True)
    t = Variable(t, volatile=True)
    x, t = x.unsqueeze(0), t.unsqueeze(0)
    output = model(x)
    output = output.squeeze(0)
    output = output.data
    x, t = x.squeeze(0), t.squeeze(0)
    current = np.multiply(x, max_values)
    output = np.multiply(output, max_values)
    target = np.multiply(t, max_values)

    print(current)
    print(target)
    print(output)

def test(station, valid_dataset, max_values, end_time, start):
    feature_dim = 6
    model = BaseModel(feature_dim, use_length, rnn_hid_dim)
    model.load_state_dict(torch.load('./files/params_'+station+'.pkl'))
    x, t = valid_dataset[start]
    count = 0
    #x, t = x.unsqueeze(0), t.unsqueeze(0)
    for i in range(end_time, 24):
        _, t = valid_dataset[start+count]
        count += 1
        x = Variable(x, volatile=True)
        x = x.unsqueeze(0)
        output = model(x)
        output = output.squeeze(0)
        output = output.data
        x = x.squeeze(0)
        x = np.vstack((x[1:], output[-1]))
        x = torch.from_numpy(x)

    acc = []
    PM25 = []
    PM10 = []
    O3 = []
    PM25_actual = []
    PM10_actual = []
    O3_actual = []
    for i in range(0, predict_time):
        _, t = valid_dataset[start+count]
        count += 1
        x = Variable(x, volatile=True)
        x = x.unsqueeze(0)
        output = model(x)
        output = output.squeeze(0)
        output = output.data
        x = x.squeeze(0)
        x = np.vstack((x[1:], output[-1]))
        x = torch.from_numpy(x)
        out = output[-1].numpy()
        tar = t[-1].numpy()
        PM25.append(out[0])
        PM10.append(out[1])
        O3.append(out[2])
        PM25_actual.append(tar[0])
        PM10_actual.append(tar[1])
        O3_actual.append(tar[2])
    output = np.multiply(x, max_values)
    t = Variable(t, volatile=True)
    target = np.multiply(t, max_values)

    acc.append(smape(PM25, PM25_actual))
    acc.append(smape(PM10, PM10_actual))
    acc.append(smape(O3, O3_actual))
    print(acc)

def predict(station, test_data, max_values, end_time, type):
    if type == 0:
        feature_dim = 6
    else:
        feature_dim = 3
    model = BaseModel(feature_dim, use_length, rnn_hid_dim)
    model.load_state_dict(torch.load('./files/params_'+station+'.pkl'))
    x = test_data
    #x, t = x.unsqueeze(0), t.unsqueeze(0)
    for i in range(end_time, 24):
        x = Variable(x, volatile=True)
        x = x.unsqueeze(0)
        output = model(x)
        output = output.squeeze(0)
        output = output.data
        x = x.squeeze(0)
        x = np.vstack((x[1:], output[-1]))
        x = torch.from_numpy(x)
    PM25 = []
    PM10 = []
    O3 = []
    for i in range(0, predict_time):
        x = Variable(x, volatile=True)
        x = x.unsqueeze(0)
        output = model(x)
        output = output.squeeze(0)
        output = output.data
        x = x.squeeze(0)
        x = np.vstack((x[1:], output[-1]))
        x = torch.from_numpy(x)
        out = output[-1].numpy()
        out = np.multiply(out, max_values)
        if (out[0] <= 0):
            out[0] = 0
        if (out[1] <= 0):
            out[1] = 0
        if (out[2] <= 0):
            out[2] = 0
        PM25.append(out[0])
        PM10.append(out[1])
        if type == 0:
            O3.append(out[2])
        else:
            O3.append(0)
    output_dict = {}
    output_dict['station_id'] = station
    output_dict['PM2.5'] = PM25
    output_dict['PM10'] = PM10
    output_dict['O3'] = O3
    if type == 0:
        beijing_output.append(output_dict)
    else:
        london_output.append(output_dict)


def make_test_data(name, length=24):
    print('> reading %s ...' % name)
    aq_matrix = np.load(open(name, 'rb'))
    total_count, dim = aq_matrix.shape
    print('> get matrix of shape', total_count, dim)

    max_values = np.max(aq_matrix, axis=0)
    aq_matrix = aq_matrix / max_values

    len, _ = aq_matrix.shape
    test_dataset = aq_matrix[(len-length):len, :]
    test_dataset = torch.from_numpy(test_dataset).float()
    return test_dataset, max_values


if __name__ == '__main__':

    for station in beijing_station:
        test_data, max_values = make_test_data('./files/test/bj_aq_test_'+station+'.npy', use_length)
        print (test_data.shape)
        predict(station, test_data, max_values, 24, 0)
    for station in london_station:
        test_data, max_values = make_test_data('./files/test/ld_aq_test_'+station+'.npy', use_length)
        predict(station, test_data, max_values, 24, 1)
    out_file = open('beijing_output.json', 'w')
    out_file.write(json.dumps(beijing_output))
    out_file.close()
    out_file = open('london_output.json', 'w')
    out_file.write(json.dumps(london_output))
    out_file.close()
    '''
    station = 'LW2'
    filename = './files/test/ld_aq_test_'+station+'.npy'
    test_data, max_values = make_test_data(filename, use_length)
    predict(station, test_data, max_values, 24, 1)
    station = 'dongsi'
    filename = './files/test/bj_aq_test_'+station+'.npy'
    test_data, max_values = make_test_data(filename, use_length)
    print (np.multiply(test_data, max_values))
    predict(station, test_data, max_values, 24, 0)

    '''
