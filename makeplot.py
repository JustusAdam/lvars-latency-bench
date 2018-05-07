import glob
import re
import json
import numpy
import matplotlib.pyplot as plt
import itertools
import sys
from scipy.stats.kde import gaussian_kde

FNAME_PATTERN = re.compile("results-(\w+)-(\d+)")

#plt.style.use("ggplot")

def dmap(f, d):
    return { k : f(v) for k, v in d.items() }

def load_file(fname):
    with open(fname) as f:
        return process_open_data_file(f)

def process_open_data_file(f):
        return list(map(int,json.load(f)))

def interpret_file_name(fname):

    match = FNAME_PATTERN.match(fname)

    return (match.group(1), match.group(2))

def get_data(arguments):
    if arguments.use_zipped_data is False:
        return dmap(load_file, get_latest_files())
    else:
        import zipfile
        with zipfile.ZipFile(arguments.use_zipped_data or 'results.zip', mode='r') as zf:
            return { ty.rstrip(".json") : process_open_data_file(zf.open(ty,mode='r')) for ty in zf.namelist() }


def get_latest_files():
    result_files = glob.glob("results-*")
    interpreted = zip(result_files, map(interpret_file_name, result_files))

    typed_results = {}

    for (fname, (exp_type, ts)) in interpreted:
        typed_results.setdefault(exp_type, []).append((fname, ts))

    return dmap(lambda vs:max(vs, key=lambda a: a[1])[0], typed_results)

def zip_latest_files(arguments):
    import zipfile

    fname = arguments.output or 'results.zip'
    
    with zipfile.ZipFile(fname, mode='w') as z:
        for ty, data_file in get_latest_files().items():
            z.write(data_file, ty + '.json')
            

def plot_data(arguments):
    d = get_data(arguments)
    save_location = arguments.output
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help="Where to write the graph or the zip file to", default=None)

    sp = parser.add_subparsers()

    plot_parser = sp.add_parser('plot')
    plot_parser.add_argument('--use-zipped-data', nargs='?', default=False, help="Use the data files from the results zip file")
    plot_parser.set_defaults(func=plot_data)

    z_parser = sp.add_parser('zip')
    z_parser.set_defaults(func=zip_latest_files)

    res = parser.parse_args()
    
    res.func(res)


if __name__ == '__main__':
    main()
