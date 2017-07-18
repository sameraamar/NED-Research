import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

def show_feature(data1, data2, title, field):
    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=16)


    a_heights, a_bins = np.histogram(data1[field])
    b_heights, b_bins = np.histogram(data2[field], bins=a_bins)

    width = (a_bins[1] - a_bins[0])/3

    ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue', label="Events")
    ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen', label="No-Events")

    ax.set_ylabel("Frequency")
    ax.set_xlabel(field)

    ax.legend(loc="upper right")

    descriptive_stat("Event:", data1[field])
    descriptive_stat("No-Event:", data2[field])


    plt.show()


    #df4 = pd.DataFrame({'Events': data1['entropy'], 'No-Events': data2['entropy']}, columns=['Events', 'No-Events'])
    #df4.plot(kind='hist', alpha=0.5)
    #plt.show()

def descriptive_stat(title, s):
    print("------------------------", title)
    print("Mean : {0:8.6f}".format(s.mean()))
    print("Minimum : {0:8.6f}".format(s.min()))
    print("Maximum : {0:8.6f}".format(s.max()))
    print("Variance : {0:8.6f}".format(s.var()))
    print("Std. deviation : {0:8.6f}".format(s.std()))


if __name__ == "__main__":
    start = datetime.datetime.now()

    #filename = 'c:/temp/features_tweet.csv'
    filename = 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/features.csv'

    data = pd.read_csv(filename, sep=',', index_col = 0)

    data1 = data[data['is_event'] == 1 ]
    data2 = data[data['is_event'] == 0 ]

    data1 = data1.head(500)
    data2 = data2.head(500)

    end = datetime.datetime.now()
    print(start, end)
    print(end-start)

    show_feature(data1, data2, "Entropy feature", 'entropy')
    show_feature(data1, data2, "Size feature", 'size')
    show_feature(data1, data2, "Average time feature", 'avg_time')

