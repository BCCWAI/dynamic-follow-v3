import json
import matplotlib.pyplot as plt
import pyperclip
import os
os.chdir("D:\Resilio Sync\df\knackerbrot-HOLDEN ASTRA RS-V BK 2017")
driving_data=[]
for filename in os.listdir("D:\Resilio Sync\df\knackerbrot-HOLDEN ASTRA RS-V BK 2017"):
    started = False
    if filename!="df-data.12550" and filename!="df-data.45121" and filename!="df-data.97447": # exclude a_rel data
        with open(filename, "r") as f:
            d_data = f.read().split("\n")
        
        d_data = [json.loads(i) for i in d_data if i != ""]
        for line in d_data:
            if not started and line[0] != 0.0:
                started=True
            
            if started:
                line[0] = max(round(line[0], 12), 0.0)
                line[1] = round(line[1], 12) #a_ego
                line[2] = max(round(line[2], 12), 0.0)
                line[3] = max(round(line[3], 12), 0.0)
                line[4] = round(line[4], 12) #a_lead
                
                driving_data.append(line)

                
#test_set=driving_data[61500:61775]
start=61500
end=62500
test_set=driving_data[start:end]
x=[i[:5] for i in test_set]
y = [i[5] - i[6] for i in test_set]

save=True
if save:
    with open("C:\Git\dynamic-follow-tf\data\\x", "w") as f:
        json.dump(x, f)
    
    with open("C:\Git\dynamic-follow-tf\data\\y", "w") as f:
        json.dump(y, f)
    

y=[i[0]*2.23694 for i in driving_data][start:end]
y2=[i[5] - i[6] for i in driving_data][start:end]
x=range(len(y))
#plt.plot(x,y)
plt.plot(x,y2)
plt.show()