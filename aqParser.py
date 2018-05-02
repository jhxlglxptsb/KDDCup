import pandas as pd
import os, json, sys
abspath = os.path.dirname(os.path.abspath(__file__))

beijing_aq_files = ['beijing_17_18_aq.csv', 'beijing_201802_201803_aq.csv']
beijing_aq_columns = {'stationId': 'station_id', 'utc_time': 'time', 'PM2.5': 'PM2.5', 'PM10': 'PM10', 'NO2': 'NO2', 'CO': 'CO', 'O3': 'O3', 'SO2': 'SO2'}
london_aq_files = ['London_historical_aqi_forecast_stations_20180331.csv', 'London_historical_aqi_other_stations_20180331.csv']
london_aq_columns = {'MeasurementDateGMT': 'time', 'station_id': 'station_id', 'PM2.5 (ug/m3)': 'PM2.5', 'PM10 (ug/m3)': 'PM10', 'NO2 (ug/m3)': 'NO2', 'Station_ID': 'station_id'}
beijing_test_files = ['beijing_201802_201803_aq.csv']

aq_files = []
aq_columns = {}

if len(sys.argv) < 2:
    print('usage: python aqParser.py beijing/london')
    exit(0)
aq_name = sys.argv[1].lower()
if aq_name == 'beijing':
    aq_files = beijing_aq_files
    aq_columns = beijing_aq_columns
elif aq_name == 'london':
    aq_files = london_aq_files
    aq_columns = london_aq_columns
elif aq_name == ''
else:
    print('usage: python aqParser.py beijing/london')
    exit(0)

aq_data = []

for aq_file in aq_files:
    f = open(os.path.join(abspath, aq_file), 'r')
    data = pd.read_csv(f)
    for index, row in data.iterrows():
        the_aq = {}
        for column_name in data.columns:
            if 'Unnamed' in column_name:
                continue
            the_aq[aq_columns[column_name]] = row[column_name]
            if aq_columns[column_name] == "station_id" and isinstance(row[column_name], str):
                if row[column_name][-3:] == "_aq":
                    the_aq[aq_columns[column_name]] = row[column_name][:-3]
        aq_data.append(the_aq)
    f.close()

out_file = open(os.path.join(abspath, aq_name + '_aq.json'), 'w')
out_file.write(json.dumps(aq_data))
out_file.close()
