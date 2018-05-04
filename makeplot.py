import glob
import re
import json
import numpy
import matplotlib.pyplot as plt
import itertools
import sys
from scipy.stats.kde import gaussian_kde

FNAME_PATTERN = re.compile("results-(\w+)-(\d+)")

plt.style.use("ggplot")

def dmap(f, d):
    return { k : f(v) for k, v in d.items() }

def load_file(fname):
    with open(fname) as f:
        return list(map(int,json.load(f)))

def interpret_file_name(fname):

    match = FNAME_PATTERN.match(fname)

    return (match.group(1), match.group(2))

def get_data():
    result_files = glob.glob("results-*")
    interpreted = zip(result_files, map(interpret_file_name, result_files))

    typed_results = {}

    for (fname, (exp_type, ts)) in interpreted:
        typed_results.setdefault(exp_type, []).append((fname, ts))

    latest_files = dmap(lambda vs:max(vs, key=lambda a: a[1])[0], typed_results)

    return dmap(load_file, latest_files)

def plot_data(d, save_location=None):
    max_len = max(map(lambda d0 : max(d0) - min(d0), d.values()))
    samplewidth = max_len / 200

    print samplewidth
    
    for ty, data in d.items():
        max_point = max(data)
        min_point = min(data)
        def bucket_num(i):
            return (i - min_point) / samplewidth
        print ty, len(data)
        frequencies = { b_num : len(list(items)) for (b_num, items) in itertools.groupby(sorted(data, key=bucket_num), bucket_num) }
        print max(frequencies.values())
        x = numpy.array(range(bucket_num(max_point)))
        y = numpy.array(map(lambda i : frequencies.get(i, 0), x))
        
        #print (len(x),len(y))
        plt.plot(x, y, label=ty)
    plt.legend()
    if save_location is None:
        plt.show()
    else:
        plt.savefig(save_location)


if __name__ == '__main__':
    _, args = sys.argv
    save_location = args[0] if len(args) > 0 else None
    plot_data(get_data(), save_location)
        
