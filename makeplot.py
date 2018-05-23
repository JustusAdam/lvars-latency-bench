from __future__ import print_function
import glob
import re
import json
import itertools
import sys
import math
import subprocess as sp
from collections import MutableMapping
from fractions import Fraction

FNAME_PATTERN = re.compile("results-(\w+)-(\d+)")

DEFAULT_EXPERIMENTS = {
    'fbm' : 'ohua-fbm',
    'sbfm' : 'ohua-sbfm',
    'sbfmpar' : 'ohua-sbfm-par',
    'LVars' : 'LVar',
    'par' : 'monad-par',
    'strategies' : 'strategies',
    'sequential' : 'sequential'
}

RT_FILE = 'res-avg-rt.json'

#plt.style.use("ggplot")

fst = lambda a : a[0]
snd = lambda a : a[1]
const = lambda a : lambda b : a

SET_SIZE = 119964

class ChainMap(MutableMapping):
    ''' A ChainMap groups multiple dicts (or other mappings) together
    to create a single, updateable view.
    The underlying mappings are stored in a list.  That list is public and can
    be accessed or updated using the *maps* attribute.  There is no other
    state.
    Lookups search the underlying mappings successively until a key is found.
    In contrast, writes, updates, and deletions only operate on the first
    mapping.
    '''

    def __init__(self, *maps):
        '''Initialize a ChainMap by setting *maps* to the given mappings.
        If no mappings are provided, a single empty dictionary is used.
        '''
        self.maps = list(maps) or [{}]          # always at least one map

    def __missing__(self, key):
        raise KeyError(key)

    def __getitem__(self, key):
        for mapping in self.maps:
            try:
                return mapping[key]             # can't use 'key in mapping' with defaultdict
            except KeyError:
                pass
        return self.__missing__(key)            # support subclasses that define __missing__

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __len__(self):
        return len(set().union(*self.maps))     # reuses stored hash values if possible

    def __iter__(self):
        return iter(set().union(*self.maps))

    def __contains__(self, key):
        return any(key in m for m in self.maps)

    def __bool__(self):
        return any(self.maps)

    def __repr__(self):
        return '{0.__class__.__name__}({1})'.format(
            self, ', '.join(map(repr, self.maps)))

    @classmethod
    def fromkeys(cls, iterable, *args):
        'Create a ChainMap with a single dict created from the iterable.'
        return cls(dict.fromkeys(iterable, *args))

    def copy(self):
        'New ChainMap or subclass with a new copy of maps[0] and refs to maps[1:]'
        return self.__class__(self.maps[0].copy(), *self.maps[1:])

    __copy__ = copy

    def new_child(self, m=None):                # like Django's Context.push()
        '''New ChainMap with a new map followed by all previous maps.
        If no map is provided, an empty dict is used.
        '''
        if m is None:
            m = {}
        return self.__class__(m, *self.maps)

    @property
    def parents(self):                          # like Django's Context.pop()
        'New ChainMap from maps[1:].'
        return self.__class__(*self.maps[1:])

    def __setitem__(self, key, value):
        self.maps[0][key] = value

    def __delitem__(self, key):
        try:
            del self.maps[0][key]
        except KeyError:
            raise KeyError('Key not found in the first mapping: {!r}'.format(key))

    def popitem(self):
        'Remove and return an item pair from maps[0]. Raise KeyError is maps[0] is empty.'
        try:
            return self.maps[0].popitem()
        except KeyError:
            raise KeyError('No keys found in the first mapping.')

    def pop(self, key, *args):
        'Remove *key* from maps[0] and return its value. Raise KeyError if *key* not in maps[0].'
        try:
            return self.maps[0].pop(key, *args)
        except KeyError:
            raise KeyError('Key not found in the first mapping: {!r}'.format(key))

    def clear(self):
        'Clear maps[0], leaving maps[1:] intact.'
        self.maps[0].clear()


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def unzip(iterable):
    l1 = []
    l2 = []
    for a, b in iterable:
        l1.append(a)
        l2.append(b)
    return (l1, l2)

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

def get_latest_n_files_of(ty, n):
    return glob.glob('results-' + ty + '-')

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

def get_runtime(file):
    d = load_file(file)['data']
    return d['finish'] - d['start']

load_runtimes = lambda files : list(map(get_runtime, files))


def avg_runtime(files):
    rts = load_runtimes(files)
    return long(sum(rts)) / long(len(rts))


def run_repeatable(arguments):

    experiments = arguments.select
    get_data = load_runtimes if arguments.no_average else avg_runtime

    sp.call(['stack', 'build'])

    def run_for_work(producer_work, consumer_work, cores):
        pwrk = str(producer_work)
        cwrk = str(consumer_work)
        depth = str(arguments.depth)
        cores = str(cores)
        reps = arguments.repetitions
        for e in experiments:
            executable = DEFAULT_EXPERIMENTS[e] + '-latency'
            for i in range(reps):
                eprint("Running {0} with {1} producer work and {2} consumer work on {4} cores, repetition {3}".format(e, pwrk, cwrk, i, cores))
                sp.check_call(['stack', 'exec', '--', executable, arguments.graph, depth, pwrk, cwrk, '+RTS', '-N' + cores])

        files = dselect(experiments, get_latest_n_files(reps))
        data = dmap(get_data, files)
        for fs in files.values():
            for f in fs:
                os.remove(f)
        return data

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

def run_config(cfg):
    e = cfg['experiment']
    pwrk = str(cfg['work']['producer'])
    cwrk = str(cfg['work']['consumer'])
    cores = str(cfg['cores'])
    reps = cfg['repetitions']
    depth = str(cfg['depth'])
    graph = cfg['graph']

    executable = cfg.get('executable', DEFAULT_EXPERIMENTS[e] + '-latency')

    for i in range(reps):
        eprint("Running {0} with {1} producer work and {2} consumer work on {4} cores, repetition {3}".format(e, pwrk, cwrk, i, cores))
        sp.check_call(['stack', 'exec', '--', executable, graph, depth, pwrk, cwrk, '+RTS', '-N' + cores])

    files = get_latest_n_files_for(e,reps)

    def load_data(f):
        with open(f, mode='r') as fp:
            return json.load(fp)

    results = [load_data(f) for f in files]

    for f in files:
        os.remove(f)

    return results

DEFAULT_CONFIG = {
    'cores' : 7,
    'repetitions': 1,
    'depth': 20,
    'work': {
        'producer': 800,
        'consumer': 800
    }
}
def run_configs(arguments):
    configs = None

    with open(arguments.config_file, mode='r') as fp:
        configs = json.load(fp)

    add_graph = { 'graph': arguments.graph }
    results = [ {'config' : cfg, 'data' : run_config(ChainMap(cfg, DEFAULT_CONFIG, add_graph))} for cfg in configs ]

    out = arguments.output if arguments.output is not None else 'results.json'

    with open(out, mode='w') as f:
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

def div(a, b):
    return Fraction(a, b)

def rel(a, b):

    if a >= b:
        return div(a, b) - 1.0
    else:
        return 1.0 - div(b, a)

def plot_rts(arguments):
    import numpy
    import matplotlib.pyplot as plt

    d = None

    plotargs = {}

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    if arguments.linestyle is not None:
        plotargs['linestyle'] = arguments.linestyle
    if arguments.marker is not None:
        plotargs['marker'] = arguments.marker

    with open(RT_FILE, mode='r') as f:
        d = json.load(f)

    for ty, d in d.items():

        (ks, vs) = unzip(d)

        x = numpy.array(map(lambda (a, b) : rel(b, a), ks))

        y = numpy.array(map(lambda v : div(SET_SIZE, v), vs))
        #y = numpy.array(map(lambda v : v, vs))
        ax.plot(x, y, label=ty, **plotargs)
#    ax.set_xlim([min(x), max(x)])
    if not arguments.no_legend:
        plt.legend()
    if arguments.log_scale:
        ax.set_xscale('log')
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
    run_parser.add_argument('--no-average', default=False, action='store_true')
    run_parser.set_defaults(func=run_repeatable)
    rt_plot_parser = sp.add_parser('plot-rt')
    rt_plot_parser.add_argument('--marker', default=None)
    rt_plot_parser.add_argument('--linestyle', default=None)
    rt_plot_parser.add_argument('--no-legend', default=False,  action='store_true')
    rt_plot_parser.add_argument('--log-scale', action='store_true')

    rt_plot_parser.set_defaults(func=plot_rts)

    dcp = sp.add_parser('run-conf')
    dcp.add_argument('graph')
    dcp.add_argument('config_file')

    dcp.set_defaults(func=run_configs)

    res = parser.parse_args()

    res.func(res)


if __name__ == '__main__':
    main()
