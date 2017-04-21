import re
from pprint import pprint
import pandas

filename = "c:/temp/threads 2m.txt"
labeled_file = "c:/data/events_db/petrovic/relevance_judgments_00000000 - flat.csv"


#%% Load judgment info

file = open (labeled_file)

labeled = []
for line in file:
    line = line.strip()
    if line == "":
        continue

    vals = line.split(",")

    #Example: 89007237128921088,19,Loaded
    row = {"id" : vals[0] , "topic" : vals[1], "status" : vals[2]}
    labeled.append( row)

print("positive labeled: " )
pprint(labeled)

file.close()

#%% Load dataset

file = open(filename)

clusters = {}
df1 = []


count = 0
members = []
for line in file:
    line = line.strip()
    count+=1

    if line=="":
        clusters[lead]["members"] = members
        members = []

    match = re.search(r"LEAD: (\d+) SIZE: (\d+) Entropy: (\d+.\d+) Age: (\d+) \(s\)", line)
    if match:
        lead = match.group(1)
        size = match.group(2)
        entropy = match.group(3)
        age = match.group(4)

        cluster = {"lead": lead, "size": size, "entropy": entropy, "age": age }
        clusters[lead] = cluster

    match = re.search(r"(\d+)	(null|\d+)	([-]?\d+.\d+)	", line)
    if match:
        members.append(match.group(1))

        df1.append ([ lead, int(size), float(entropy), int(age), match.group(1) ])
        if count%1000 == 0:
            #tmp = pandas.DataFrame(df1, columns=["lead", "size", "entropy", "age", "member"])

            #print(tmp)
            #df.append( tmp)
            #df1=[]
            print(count)



file.close()

print("# clusters:  ", len(clusters))

df = pandas.DataFrame(df1, columns=["lead", "size", "entropy", "age", "member"])
del df1
pprint(df)

#analyze

print('looking for known (labeled) tweets ...')
for tid in labeled:
    #print (tid['id'])
    tmp = df [ df["member"] == tid['id'] ]
    if(len(tmp) > 0):
        print ( tmp )

print(len(df))