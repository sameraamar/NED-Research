import re
from pprint import pprint
import pandas


try:
    from pandasql import sqldf #, load_meat, load_births
except:
    print("try to install using:\npip install pandasql")
    exit()


VERSION = "V1"
SUFFEX = "300k"
FOLDER = "c:/temp"
sep = ","



filename = FOLDER + "/" + "threads_" + SUFFEX + "_" + VERSION + ".txt" #;""c:/temp/threads 2m.txt"
labeled_file = "c:/data/events_db/petrovic/relevance_judgments_00000000 - flat.csv"

#%% Load judgment info

labeled_ds = pandas.read_csv(labeled_file, sep=sep)
print("labeled dataset: " , len(labeled_ds))
print(labeled_ds.head())

#%% Load dataset

file = open(filename)

clusters = []
memebers = []

count = 0
#members = []
for line in file:
    line = line.strip()
    count+=1

    match = re.search(r"LEAD: (\d+) SIZE: (\d+) Entropy: (\d+.\d+) Age: (\d+) \(s\)", line)
    if match:
        lead = match.group(1)
        size = match.group(2)
        entropy = match.group(3)
        age = match.group(4)


    #match = re.search(r"(\d+)	(null|\d+)	([-]?\d+.\d+)	", line)
    match = re.search(r"(\d+)	[^	]+	(\d+)	(null|\d+)	([-]?\d+.\d+)	", line)
    if match:
        #members.append(match.group(1))

        clusters.append ([ lead, int(size), float(entropy), int(age), match.group(1) ])
        if count%1000 == 0:
            #tmp = pandas.DataFrame(df1, columns=["lead", "size", "entropy", "age", "member"])

            #print(tmp)
            #df.append( tmp)
            #clusters=[]
            print(count)



file.close()

df = pandas.DataFrame(clusters, columns=["lead", "size", "entropy", "age", "member"])
del clusters
print(df.tail(50))

print(len(df))

df.to_csv("c:/temp/threads.csv", index_label="#")



#analyze

events_no = sqldf("SELECT COUNT(*) FROM df ;", locals())
print(events_no)

#print('looking for known (labeled) tweets ...')
#for tid in labeled:
#    #print (tid['id'])
#    tmp = df [ df["member"] == tid['id'] ]
#    if(len(tmp) > 0):
#        print ( tmp )

