from matplotlib import pyplot as plt
import json
import matplotlib.patches as mpatches

b_station_file = open("beijing_station.json", "r")
b_station_data = json.loads(b_station_file.read())
plt.figure(0)
plt.figure(figsize = (20, 15))
bx = [[],[],[],[]]
by = [[],[],[],[]]
bname = [[],[],[],[]]
bcolor = ['k', 'b', 'y', 'r']
b_station_type = {"city":0, "countryside":1, "control":2, "traffic":3}
b_type = ["city", "countryside", "control", "traffic"]
for b_station in b_station_data:
    bx[b_station_type[b_station["area"]]].append(b_station["longitude"])
    by[b_station_type[b_station["area"]]].append(b_station["latitude"])
    bname[b_station_type[b_station["area"]]].append(b_station["station_id"])
for i in range(4):
    plt.scatter(bx[i],by[i],c=bcolor[i],label=b_type[i])
    '''
    for j in range(len(bname[i])):
        plt.annotate(s=bname[i][j], xy=(bx[i][j],by[i][j]), xytext=(-20, 10), textcoords='offset points')
    '''
plt.legend(bbox_to_anchor=(1,1), loc='center left')

#plt.show()
plt.savefig("beijing_station_without_name.png")
plt.close(0)
