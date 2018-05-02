import pandas as pd
import os, json, sys
abspath = os.path.dirname(os.path.abspath(__file__))

beijing_meo_grid_files = ['Beijing_historical_meo_grid.csv']
beijing_meo_grid_columns = {'stationName': 'station_name', 'utc_time': 'time', 'latitude': 'latitude', 'longitude': 'longitude', 'temperature': 'temperature', 'pressure': 'pressure', 'humidity': 'humidity', 'wind_direction': 'wind_direction', 'wind_speed/kph': 'wind_speed'}
london_meo_grid_files = ['London_historical_meo_grid.csv']
london_meo_grid_columns = {'stationName': 'station_name', 'utc_time': 'time', 'latitude': 'latitude', 'longitude': 'longitude', 'temperature': 'temperature', 'pressure': 'pressure', 'humidity': 'humidity', 'wind_direction': 'wind_direction', 'wind_speed/kph': 'wind_speed'}

meo_grid_files = []
meo_grid_columns = {}

if len(sys.argv) < 2:
    print('usage: python meoGridParser.py beijing/london')
    exit(0)
meo_grid_name = sys.argv[1].lower()
if meo_grid_name == 'beijing':
    meo_grid_files = beijing_meo_grid_files
    meo_grid_columns = beijing_meo_grid_columns
elif meo_grid_name == 'london':
    meo_grid_files = london_meo_grid_files
    meo_grid_columns = london_meo_grid_columns
else:
    print('usage: python meoGridParser.py beijing/london')
    exit(0)

meo_grid_data = []

for meo_grid_file in meo_grid_files:
    f = open(os.path.join(abspath, meo_grid_file), 'r')
    data = pd.read_csv(f)
    i = 0
    for index, row in data.iterrows():
        i += 1
        if i >= 1000:
            break
        the_meo_grid = {}
        for column_name in data.columns:
            if 'Unnamed' in column_name:
                continue
            the_meo_grid[meo_grid_columns[column_name]] = row[column_name]
        meo_grid_data.append(the_meo_grid)
    f.close()

out_file = open(os.path.join(abspath, meo_grid_name + '_meo_grid.json'), 'w')
out_file.write(json.dumps(meo_grid_data))
out_file.close()
