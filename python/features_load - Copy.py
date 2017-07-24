import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import datetime


import scipy.stats as stats
import matplotlib.mlab as mlab

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def show_feature(ax, data, binstitle, field):

    a_heights, a_bins = np.histogram(data1[field])
    b_heights, b_bins = np.histogram(data2[field], bins=a_bins)

    width = (a_bins[1] - a_bins[0])/3

    ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue', label="Events")
    ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen', label="No-Events")

    ax.set_ylabel("Frequency")
    ax.set_xlabel(field)

    ax.legend(loc="upper right")

    print('--------------------------', title + " (" + field + ")")
    descriptive_stat("Event", data1[field])
    print('----')
    descriptive_stat("No-Event", data2[field])

    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    #formatter = FuncFormatter(to_percent)

    # Set the formatter
    #plt.gca().yaxis.set_major_formatter(formatter)

    x = data1[field]

    density = stats.gaussian_kde(x)
    plt.plot(a_bins[:-1], density(a_bins[:-1]), 'b--')

    #y = mlab.normpdf(a_bins, x.mean(), x.var())
    #plt.plot(a_bins, y, 'b--')


def show_two_features(data1, data2, title, field):
    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=16)

    show_feature(data1, title, field)

    plt.show()

def descriptive_stat(title, s):
    print(" >> " + title + " << ")
    print("data size : {0:8d}".format(len(s)))
    print("Mean : {0:8.6f}".format(s.mean()))
    print("Minimum : {0:8.6f}".format(s.min()))
    print("Maximum : {0:8.6f}".format(s.max()))
    print("Variance : {0:8.6f}".format(s.var()))
    print("Std. deviation : {0:8.6f}".format(s.std()))


def run_feature_analysis(data, field_titles, field_names, is_event_field='is_event', sep=',', sample_size=-1):
    start = datetime.datetime.now()

    data1 = data[data[is_event_field] == 1 ]
    data2 = data[data[is_event_field] == 0 ]

    if sample_size>0:
        sample_size = min(sample_size, len(data1))
        sample_size = min(sample_size, len(data2))

        data1 = data1.sample(sample_size) #.head(sample_size)
        data2 = data2.sample(sample_size) #.head(sample_size)

    end = datetime.datetime.now()

    for i in range(len(field_names)):
        show_feature(data1, data2, field_titles[i], field_names[i])

    print(start, end)
    print(end-start)

if __name__ == "__main__":
    # folder = 'c:/temp/features_tweet.csv'
    folder = 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/'

    data = pd.read_csv(folder + 'features.csv', sep=',', index_col=0)
    run_feature_analysis(data, ["Entropy feature", "Size feature", "Average time feature"], ['entropy', 'size', 'avg_time'], sample_size=500)

    data = pd.read_csv(folder + 'tree_features.csv', sep=',', index_col=0)
    run_feature_analysis(data, ["Tree Size", "retweet count", 'likes average'], ['tree_size', 'rtwt_count', 'likes_avg'], is_event_field='is_tree_event', sample_size=500)

