from matplotlib import pyplot as plt
import json

import time
import datetime

b_meo = json.load(open('beijing_meo.json'))

def adjust_time_b(x):
    dt_obj = datetime.datetime.strptime(x['time'], '%Y-%m-%d %H:%M:%S')
    ts = time.mktime(dt_obj.timetuple())
    x['time'] = ts
    return x

def b_paint_meo(station):
    meo = [x for x in b_meo if x['station_id'] == station]
    meo = [adjust_time_b(x) for x in meo]
    meo = map(dict, set(tuple(sorted(x.items())) for x in meo))
    meo = sorted(meo, key=lambda a: a['time'])

    features = ['temperature','pressure','humidity','wind_speed','wind_direction']

    plt.figure(0)
    plt.figure(figsize = (25, 20))
    for j, feat in enumerate(features):
        x = []
        y = []
        for i, m in enumerate(meo):
            if feat == 'wind_direction' and meo[i][feat] == 999017:
                continue
            else:
                x.append(meo[i]['time'])
                y.append(meo[i][feat])
        plt.subplot(5,1,j+1)
        plt.plot(x,y)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title(feat)
        plt.subplots_adjust(top=0.95)
    plt.savefig("./picture/beijing_meo/meo_"+station+".png")
    plt.close(0)


b_t = set()
for x in b_meo:
    station = x['station_id']
    b_t.add(station)
for station in b_t:
    b_paint_meo(station)
