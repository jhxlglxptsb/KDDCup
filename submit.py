import time
# coding: utf-8
london_stations = ['CD1', 'BL0', 'GR4', 'MY7', 'HV1', 'GN3', 'GR9', 'LW2', 'GN0', 'KF1', 'CD9', 'ST5', 'TH4']

import requests, sys, json
import pandas as pd

username = ''
team_token = ''
try:
    config_file = open('config.json', 'r')
    config_struct = json.loads(config_file.read())
    username = config_struct['username']
    team_token = config_struct['team_token']
    config_file.close()
except:
    print('try to get default config failure, please input your username (you can get your username on top right when you login kddcup index) and team_token (you can get it from your team leader\'s my_team page)')
    username = raw_input('username:')
    team_token = raw_input('team_token:')
    config_file = open('config.json', 'w')
    config_file.write(json.dumps({'username' : username, 'team_token' : team_token}))
    config_file.close()

print('username:' + username)
print('team_token:' + team_token)
print('You can change your default config in config.json')

if len(sys.argv) < 3:
    print('Usage: python submit.py [beijing_jsonfile] [london_jsonfile]')
    exit(0)


try:
    bjjson_text = open(sys.argv[1]).read()
except:
    print('beijing_jsonfile open failure')
    exit(0)

try:
    ldjson_text = open(sys.argv[2]).read()
except:
    print('london_jsonfile open failure')
    exit(0)

test_ids = []
pm25s = []
pm10s = []
o3s = []

beijing_datas = json.loads(bjjson_text)
if len(beijing_datas) != 35:
    print('station number of beijing is not 35')
    exit(0)

for data in beijing_datas:
    for i in range(48):
        sta = data['station_id']
        if len(sta) > 10:
            sta = sta[0:10]
        test_ids.append(sta + '_aq#' + str(i))
        pm25s.append(int(float(data['PM2.5'][i])))
        pm10s.append(int(float(data['PM10'][i])))
        o3s.append(int(float(data['O3'][i])))

london_datas = json.loads(ldjson_text)
if len(london_datas) < 13:
    print('station number of london less than 13')
    exit(0)

for data in london_datas:
    if data['station_id'] not in london_stations:
        continue
    for i in range(48):
        test_ids.append(data['station_id'] + '#' + str(i))
        pm25s.append(int(float(data['PM2.5'][i])))
        pm10s.append(int(float(data['PM10'][i])))
        o3s.append(int(float(data['O3'][i])))

column_testid = pd.Series(test_ids, name='test_id')
column_pm25 = pd.Series(pm25s, name='PM2.5')
column_pm10 = pd.Series(pm10s, name='PM10')
column_o3 = pd.Series(o3s, name='O3')
con = pd.concat([column_testid, column_pm25, column_pm10, column_o3], axis=1)

timestamp = str(int(time.time()))

con.to_csv(timestamp + 'result.csv', index = False)

files={'files': open(timestamp + 'result.csv','rb')}




data = {
    "user_id": username,   #user_id is your username which can be found on the top-right corner on our website when you logged in.
    "team_token": team_token, #your team_token.
    "description": 'submit a new result:' + timestamp + 'result.csv',  #no more than 40 chars.
    "filename": timestamp + "result.csv", #your filename
}

url = 'https://biendata.com/competition/kdd_2018_submit/'

response = requests.post(url, files=files, data=data)

print(response.text)
