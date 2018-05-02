import requests, sys, math, json
import pandas as pd

beijing_aq_columns = {'id' : 'id', 'station_id' : 'station_id', 'time' : 'time', 'PM25_Concentration' : 'PM2.5', 'PM10_Concentration' : 'PM10', 'NO2_Concentration' : 'NO2', 'CO_Concentration' : 'CO', 'O3_Concentration' : 'O3', 'SO2_Concentration' : 'SO2'}
london_aq_columns = {'id' : 'id', 'station_id' : 'station_id', 'time' : 'time', 'PM25_Concentration' : 'PM2.5', 'PM10_Concentration' : 'PM10', 'NO2_Concentration' : 'NO2', 'CO_Concentration' : 'CO', 'O3_Concentration' : 'O3', 'SO2_Concentration' : 'SO2'}

if len(sys.argv) < 3:
    print('Usage: python getdata.py [start_time(2018-04-01-0, for example)] [end_time(2018-04-01-23, for example)]')
    exit(0)

beijing_url = 'https://biendata.com/competition/airquality/bj/' + sys.argv[1] + '/' + sys.argv[2] + '/2k0d1d8'
london_url = 'https://biendata.com/competition/airquality/ld/' + sys.argv[1] + '/' + sys.argv[2] + '/2k0d1d8'

datas = []
response = requests.get(beijing_url)
aq_text = response.text.replace('\r', '').split('\n')
columns = aq_text[0].split(',')
columns = list(map(lambda x : beijing_aq_columns[x], columns))
for line in aq_text[1:]:
    words = line.split(',')
    if(len(words) != len(columns)):
        continue
    data = {}
    for i in range(len(columns)):
        if columns[i] == 'station_id':
            data[columns[i]] = words[i][:-3]
        elif columns[i] in ['id', 'time']:
            data[columns[i]] = words[i]
        else:
            data[columns[i]] = float('nan') if words[i] == '' else float(words[i])
    del data['id']
    datas.append(data)
out_file = open('beijing_aq_' + sys.argv[1] + '_' + sys.argv[2], 'w')
out_file.write(json.dumps(datas))
out_file.close()

datas = []
response = requests.get(london_url)
aq_text = response.text.replace('\r', '').split('\n')
columns = aq_text[0].split(',')
columns = list(map(lambda x : london_aq_columns[x], columns))
for line in aq_text[1:]:
    words = line.split(',')
    if(len(words) != len(columns)):
        continue
    data = {}
    for i in range(len(columns)):
        if columns[i] == 'station_id':
            data[columns[i]] = words[i]
        elif columns[i] in ['id', 'time']:
            data[columns[i]] = words[i]
        else:
            data[columns[i]] = float('nan') if words[i] == '' else float(words[i])
    del data['id']
    datas.append(data)
out_file = open('london_aq_' + sys.argv[1] + '_' + sys.argv[2], 'w')
out_file.write(json.dumps(datas))
out_file.close()
