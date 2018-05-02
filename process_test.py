from __future__ import print_function
import json
import time
import datetime

import numpy as np

def adjust_time(x):
    dt_obj = datetime.datetime.strptime(x['time'], '%Y-%m-%d %H:%M:%S')
    ts = time.mktime(dt_obj.timetuple())
    x['time'] = ts
    return x

def adjust_time_l(x):
    dt_obj = datetime.datetime.strptime(x['time'], '%Y/%m/%d %H:%M')
    ts = time.mktime(dt_obj.timetuple())
    x['time'] = ts
    return x

def process(aq, station, type):  #0for beijing, 1for london
    print('> %s', station)
    print('> get %d entries for %s' % (len(aq), station))
    aq = [adjust_time(x) for x in aq]
    aq = map(dict, set(tuple(sorted(x.items())) for x in aq))
    aq = sorted(aq, key=lambda a: a['time'])
    old = None
    count = 0
    best_missing_time = 0
    new_aq = []
    for x in aq:
        if old is not None:
            try:
                assert x['time'] - old['time'] == 3600.
            except AssertionError:
                missing_time = x['time'] - old['time']
                if missing_time > best_missing_time:
                    best_missing_time = missing_time
                # print(old)
                # print(x)
                count += 1
                # print('--------------------------------')
                c_time = old['time'] + 3600
                while c_time < x['time']:
                    new_x = x.copy()
                    new_x['time'] = c_time
                    new_aq.append(new_x)
                    c_time += 3600

        old = x
    print('> missing data count', count)
    print('> best missing time', best_missing_time)

    new_aq.extend(aq)
    new_aq = sorted(new_aq, key=lambda a: a['time'])
    aq = new_aq
    print('> finally %d entries for %s' % (len(aq), station))

    if type == 0:
        features = ['PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2']
    else:
        features = ['PM2.5', 'PM10', 'NO2']

    count = 0
    for i, x in enumerate(aq):
        for feat in features:
            if x[feat] != x[feat]:
                count += 1
                j = i + 1
                while j != len(aq) and aq[j][feat] != aq[j][feat]:
                    j += 1
                if j!= len(aq):
                    x[feat] = aq[j][feat]
                else:
                    x[feat] = 0

    print('> NaN feature count', count)

    if type == 0:
        bj_aq = np.zeros((len(aq), 6))
        for i, x in enumerate(aq):
            for j, feat in enumerate(features):
                bj_aq[i, j] = aq[i][feat]
        np.save(open('./files/test/bj_aq_test_'+station+'.npy', 'wb'), bj_aq)
    else:
        ld_aq = np.zeros((len(aq), 3))
        for i, x in enumerate(aq):
            for j, feat in enumerate(features):
                ld_aq[i, j] = aq[i][feat]
        np.save(open('./files/test/ld_aq_test_'+station+'.npy', 'wb'), ld_aq)

beijing_station = ["dongsi", "tiantan", "guanyuan", "wanshouxigong", "aotizhongxin", \
                    "nongzhanguan", "wanliu", "beibuxinqu", "zhiwuyuan", "fengtaihuayuan", \
                    "yungang", "gucheng", "fangshan", "daxing", "yizhuang", \
                    "tongzhou", "shunyi", "pingchang", "mentougou", "pinggu", \
                    "huairou", "miyun", "yanqin", "dingling", "badaling", \
                    "miyunshuiku", "donggaocun", "yongledian", "yufa", "liulihe", \
                    "qianmen", "yongdingmennei", "xizhimenbei", "nansanhuan", "dongsihuan"]
london_station = ['CD1', 'BL0', 'GR4', 'MY7', 'HV1', 'GN3', 'GR9', 'LW2', 'GN0', 'KF1', 'CD9', 'ST5', 'TH4']

aq_all = json.load(open('beijing_aq_test.json'))
print('> get %d entries' % len(aq_all))
aq_london = json.load(open('london_aq_test.json'))
print('> get %d entries' % len(aq_london))


for station in beijing_station:
    aq = [x for x in aq_all if x['station_id'] == station]
    process(aq, station, 0)

for station in london_station:
    aq = [x for x in aq_london if x['station_id'] == station]
    process(aq, station, 1)
