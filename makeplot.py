import glob
import re
import json
import numpy
import matplotlib.pyplot as plt
import itertools
import sys
from scipy.stats.kde import gaussian_kde
import math

FNAME_PATTERN = re.compile("results-(\w+)-(\d+)")

#plt.style.use("ggplot")

fst = lambda a : a[0]
snd = lambda a : a[1]

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
        return [dmap(load_file, get_latest_files())]
    else:
        import zipfile
        with zipfile.ZipFile(arguments.use_zipped_data or 'results.zip', mode='r') as zf:
            files = {}
            for ty in zf.namelist():
                ty0 = ty.rstrip('.json')
                sty = ty0.rsplit('-', 1)
                
                if len(sty) == 1:
                    ty1 = sty[0]
                    tyNum = 0
                elif len(sty) == 2:
                    ty1, tyNum = sty
                else:
                    raise Exception("invalid split: " + sty)

                files.setdefault(tyNum, {})[ty1] = process_open_data_file(zf.open(ty,mode='r'))
        
            return [files[i] for i in range(len(files))]


def get_latest_files():
    return dmap(fst, get_latest_files(1))

def get_latest_n_files(n):
    result_files = glob.glob("results-*")
    interpreted = zip(result_files, map(interpret_file_name, result_files))

    typed_results = {}

    for (fname, (exp_type, ts)) in interpreted:
        typed_results.setdefault(exp_type, []).append((fname, ts))

    return dmap(lambda vs:map(fst, sorted(vs, key=snd, reverse=True))[0:n], typed_results)

def zip_latest_files(arguments):
    import zipfile

    fname = arguments.output or 'results.zip'
    
    with zipfile.ZipFile(fname, mode='w') as z:
        for ty, data_files in get_latest_n_files(arguments.num).items():
            for i, f in zip(itertools.count(), data_files):
                z.write(data_file, ty + ('-' + n if n < 0 else '') + '.json')
            

def plot_data(arguments):
    d = get_data(arguments)
    save_location = arguments.output
    #max_len = max(map(lambda d0 : max(d0) - min(d0), d.values()))
    samplewidth = 100
    #samplewidth = max_len / arguments.buckets

    print samplewidth

    fig = plt.figure()


    plotargs = {}

    if arguments.linestyle is not None:
        plotargs['linestyle'] = arguments.linestyle
    if arguments.marker is not None:
        plotargs['marker'] = arguments.marker

    experiments = len(d)
    gridx = int(math.ceil(math.sqrt(experiments)))
    gridy= gridx

    print experiments
    print gridx

    for i in range(experiments):
        ax = fig.add_subplot(i + 1,gridx,gridy)
        for ty, data in d[i].items():
            max_point = max(data)
            min_point = min(data)
            def bucket_num(i):
                return (i - min_point) / samplewidth
            frequencies = { b_num : len(list(items))
                            for (b_num, items) in itertools.groupby(sorted(data, key=bucket_num), bucket_num) }
            x = numpy.array(range(bucket_num(max_point)))
            y = numpy.array(map(lambda i : frequencies.get(i, 0), x))
        
            #print (len(x),len(y))
            ax.plot(x[:arguments.slice_size], y[:arguments.slice_size], label=ty, **plotargs)
    if arguments.log_scale:
        ax.set_xscale('log')
    ax.legend()
    if save_location is None:
        plt.show()
    else:
        plt.savefig(save_location)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output'
                        , help="Where to write the graph or the zip file to"
                        , default=None)

    sp = parser.add_subparsers()

    plot_parser = sp.add_parser('plot')
    plot_parser.add_argument('--use-zipped-data', nargs='?'
                             , default=False
                             , help="Use the data files from the results zip file")
    plot_parser.add_argument('-b', '--buckets', type=int, default=200)
    plot_parser.add_argument('--slice-size', type=int, default=-1)
    plot_parser.add_argument('--marker', default=None)
    plot_parser.add_argument('--linestyle', default=None)
    plot_parser.add_argument('--log-scale', action='store_true')
    plot_parser.set_defaults(func=plot_data)
    z_parser = sp.add_parser('zip')
    z_parser.add_argument('-n', '--num', type=int, default=1)
    z_parser.set_defaults(func=zip_latest_files)

    res = parser.parse_args()
    
    res.func(res)


if __name__ == '__main__':
    main()
