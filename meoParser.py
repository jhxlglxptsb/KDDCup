import pandas as pd
import os, json, sys
abspath = os.path.dirname(os.path.abspath(__file__))

beijing_meo_files = ['beijing_17_18_meo.csv']
beijing_meo_columns = {'station_id': 'station_id', 'utc_time': 'time', 'latitude': 'latitude', 'longitude': 'longitude', 'temperature': 'temperature', 'pressure': 'pressure', 'humidity': 'humidity', 'wind_direction': 'wind_direction', 'wind_speed': 'wind_speed', 'weather': 'weather'}
london_meo_files = []
london_meo_columns = {'station_id': 'station_id', 'utc_time': 'time', 'latitude': 'latitude', 'longitude': 'longitude', 'temperature': 'temperature', 'pressure': 'pressure', 'humidity': 'humidity', 'wind_direction': 'wind_direction', 'wind_speed': 'wind_speed', 'weather': 'weather'}

meo_files = []
meo_columns = {}

if len(sys.argv) < 2:
    print('usage: python meoParser.py beijing/london')
    exit(0)
meo_name = sys.argv[1].lower()
if meo_name == 'beijing':
    meo_files = beijing_meo_files
    meo_columns = beijing_meo_columns
elif meo_name == 'london':
    meo_files = london_meo_files
    meo_columns = london_meo_columns
else:
    print('usage: python meoParser.py beijing/london')
    exit(0)

meo_data = []

for meo_file in meo_files:
    f = open(os.path.join(abspath, meo_file), 'r')
    data = pd.read_csv(f)
    for index, row in data.iterrows():
        the_meo = {}
        for column_name in data.columns:
            if 'Unnamed' in column_name:
                continue
            the_meo[meo_columns[column_name]] = row[column_name]
            if meo_columns[column_name] == 'station_id' and isinstance(row[column_name], str):
                if row[column_name][-4:] == '_meo':
                    the_meo[meo_columns[column_name]] = row[column_name][:-4]
        meo_data.append(the_meo)
    f.close()

out_file = open(os.path.join(abspath, meo_name + '_meo.json'), 'w')
out_file.write(json.dumps(meo_data))
out_file.close()
