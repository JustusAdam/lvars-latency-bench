import glob
import re
import json
import itertools
import sys
import math
import subprocess as sp

FNAME_PATTERN = re.compile("results-(\w+)-(\d+)")

DEFAULT_EXPERIMENTS = {
    'fbm' : 'ohua-fbm',
    'sfbm' : 'ohua-sbfm',
    'LVars' : 'LVar',
    'monad-par' : 'monad-par',
    'strategies' : 'strategies'
}

#plt.style.use("ggplot")

fst = lambda a : a[0]
snd = lambda a : a[1]
const = lambda a : lambda b : a

def dmap(f, d):
    return { k : f(v) for k, v in d.items() }

def dselect(items, d):
    allowed = frozenset(items)
    return { k : v for k,v in d.items() if k in allowed }

def load_file(fname):
    with open(fname) as f:
        return process_open_data_file(f)

def process_open_data_file(f):
    return json.load(f)

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
                    tyNum = int(tyNum)
                else:
                    raise Exception("invalid split: " + sty)

                files.setdefault(tyNum, {})[ty1] = process_open_data_file(zf.open(ty,mode='r'))
        
            return [files[i] for i in range(len(files))]


def get_latest_files():
    return dmap(fst, get_latest_n_files(1))

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

    restrictions = arguments.restrict

    filterf = (lambda a : a in frozenset(restrictions)) if restrictions is not None else const(True)
    
    with zipfile.ZipFile(fname, mode='w') as z:
        for ty, data_files in get_latest_n_files(arguments.num).items():
            if filterf(ty):
                for i, data_file in zip(itertools.count(), data_files):
                    z.write(data_file, ty + ('-' + str(i) if i > 0 else '') + '.json')

def avg_runtime(files):
    rts = []

    for f in files:
        d = load_file(f)['data']
        rts.append(d['finish'] - d['start'])

    return sum(rts) / len(rts)

RT_FILE = 'res-avg-rt.json'

def run_repeatable(arguments):

    experiments = arguments.select

    sp.call(['stack', 'build'])
    
    def run_for_work(producer_work, consumer_work):
        for e in experiments:
            for _ in range(arguments.repetitions):
                sp.check_call(['stack', 'exec', '--', DEFAULT_EXPERIMENTS[e] + '-latency', arguments.graph, str(arguments.depth), str(producer_work), str(consumer_work), '+RTS', '-N' + str(arguments.cores)])

        files = dselect(experiments, get_latest_n_files(arguments.repetitions))

        return dmap(avg_runtime, files)

    def extract_work(s):
        pw = None
        cw = None

        if ':' in s:
            pws, cws = s.split(':', 1)
            pw = int(pws)
            cw = int(cws)
        else:
            pw = int(s)
            cw = pw
        return (pw, cw)
    
    
    works = map(extract_work, arguments.work)

    results = {}

    for pw, cw in works:
        for ty, res in run_for_work(pw, cw).items():
            results.setdefault(ty, []).append(((pw, cw), res))

    with open(RT_FILE, mode='w') as f:
        json.dump(results, f)
    

def plot_data(arguments):
    import numpy
    import matplotlib.pyplot as plt
    d = get_data(arguments)
    save_location = arguments.output
    #max_len = max(map(lambda d0 : max(d0) - min(d0), d.values()))
    samplewidth = 100
    #samplewidth = max_len / arguments.buckets

    fig = plt.figure()


    plotargs = {}

    if arguments.linestyle is not None:
        plotargs['linestyle'] = arguments.linestyle
    if arguments.marker is not None:
        plotargs['marker'] = arguments.marker

    experiments = len(d)
    gridx = int(math.ceil(math.sqrt(experiments)))
    gridy= gridx

    for i in range(experiments):
        
        ax = fig.add_subplot(gridx,gridy, i + 1)

        for ty, full_data in d[i].items():
            data = full_data["data"]
            arrivals = data["arrivals"]
            max_point = max(arrivals)
            min_point = data["start"]
            def bucket_num(i):
                return (i - min_point) / samplewidth
            frequencies = { b_num : len(list(items))
                            for (b_num, items) in itertools.groupby(sorted(arrivals, key=bucket_num), bucket_num) }
            x = numpy.array(range(bucket_num(max_point)))
            y = numpy.array(map(lambda i : frequencies.get(i, 0), x))
        
            #print (len(x),len(y))
            ax.plot(x[:arguments.slice_size], y[:arguments.slice_size], label=ty, **plotargs)
    if arguments.log_scale:
        ax.set_xscale('log')
    if not arguments.no_legend:
        ax.legend()
    if save_location is None:
        plt.show()
    else:
        plt.savefig(save_location)

def unzip_dict(d):
    keys = []
    vals = []
    for k, v in d:
        keys.append(k)
        vals.append(v)
    return (keys, vals)
        
def plot_rts(arguments):
    import numpy
    import matplotlib.pyplot as plt

    d = None

    plotargs = {}

    if arguments.linestyle is not None:
        plotargs['linestyle'] = arguments.linestyle
    if arguments.marker is not None:
        plotargs['marker'] = arguments.marker
    
    with open(RT_FILE, mode='r') as f:
        d = json.load(f)

    for ty, d in d.items():
        (ks, vs) = unzip(d)
        x = numpy.array(map(lambda (a, b) : a / b, ks))
        y = numpy.array(vs)
        plt.plot(x, y, label=ty, **plotargs)

    if arguments.output is None:
        plt.show()
    else:
        plt.savefig(arguments.output)

        
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
    plot_parser.add_argument('--no-legend', default=False,  action='store_true')
    plot_parser.set_defaults(func=plot_data)
    z_parser = sp.add_parser('zip')
    z_parser.add_argument('-n', '--num', type=int, default=1)
    z_parser.add_argument('-r', '--restrict', nargs='*')
    z_parser.set_defaults(func=zip_latest_files)
    run_parser = sp.add_parser('run')
    run_parser.add_argument('-w', '--work', nargs='+')
    run_parser.add_argument('-s', '--select', nargs='*', default=DEFAULT_EXPERIMENTS)
    run_parser.add_argument('-r', '--repetitions', type=int)
    run_parser.add_argument('--depth', type=int)
    run_parser.add_argument('-c', '--cores', type=int, default=7)
    run_parser.add_argument('-g', '--graph')
    run_parser.set_defaults(func=run_repeatable)
    rt_plot_parser = sp.add_parser('plot-rt')
    rt_plot_parser.add_argument('--marker', default=None)
    rt_plot_parser.add_argument('--linestyle', default=None)

    rt_plot_parser.set_defaults(func=plot_rts)
    

    res = parser.parse_args()
    
    res.func(res)


if __name__ == '__main__':
    main()
