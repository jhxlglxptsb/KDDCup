from __future__ import print_function

from matplotlib import pyplot as plt
import json

import time
import datetime

import numpy as np

b_aq = json.load(open('beijing_aq.json'))
l_aq = json.load(open('london_aq.json'))

def adjust_time_b(x):
    dt_obj = datetime.datetime.strptime(x['time'], '%Y-%m-%d %H:%M:%S')
    ts = time.mktime(dt_obj.timetuple())
    x['time'] = ts
    return x

def adjust_time_l(x):
    dt_obj = datetime.datetime.strptime(x['time'], '%Y/%m/%d %H:%M')
    ts = time.mktime(dt_obj.timetuple())
    x['time'] = ts
    return x

def b_paint_aq(station):
    aq = [x for x in b_aq if x['station_id'] == station]
    aq = [adjust_time_b(x) for x in aq]
    aq = map(dict, set(tuple(sorted(x.items())) for x in aq))
    aq = sorted(aq, key=lambda a: a['time'])

    features = ['PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2']

    plt.figure(0)
    plt.figure(figsize = (25, 20))
    for j, feat in enumerate(features):
        x = []
        y = []
        for i, m in enumerate(aq):
            x.append(aq[i]['time'])
            y.append(aq[i][feat])
        plt.subplot(6,1,j+1)
        plt.plot(x,y)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title(feat)
        plt.subplots_adjust(top=0.95)
    plt.savefig("./picture/beijing_aq/beijing_aq_"+station+".png")
    plt.close(0)

def l_paint_aq(station):
    aq = [x for x in l_aq if x['station_id'] == station]

    aq = [adjust_time_l(x) for x in aq]
    aq = map(dict, set(tuple(sorted(x.items())) for x in aq))
    aq = sorted(aq, key=lambda a: a['time'])

    features = ['PM2.5', 'PM10', 'NO2']

    plt.figure(1)
    plt.figure(figsize = (25, 12))
    for j, feat in enumerate(features):
        x = []
        y = []
        for i, m in enumerate(aq):
            tmp = aq[i][feat]
            if tmp != tmp:
                continue
            x.append(aq[i]['time'])
            y.append(aq[i][feat])
        plt.subplot(3,1,j+1)
        plt.plot(x,y)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.subplots_adjust(top=0.95)
        plt.title(feat)
    print(station)
    plt.savefig("./picture/london_aq/london_aq_"+station+".png")
    plt.close(1)



b_t = set()
for x in b_aq:
    station = x['station_id']
    b_t.add(station)
for station in b_t:
    b_paint_aq(station)

'''
l_t = set()
for x in l_aq:
    station = x['station_id']
    if station != station:
        continue
    l_t.add(station)
for station in l_t:
    l_paint_aq(station)
'''
