import argparse
import collections
import json
import multiprocessing
import os
import subprocess
import time

import tqdm
import pandas as pd


Benchmark = collections.namedtuple(
    'Benchmark', ['file', 'name', 'ops', 'distribute', 'max_num_ops'])

benchmarks = [
    Benchmark('benchmarks/doitgen.mlir', 'doitgen',
              ['mhlo.dot_general'], False, 3),
    Benchmark('testfiles/correlation_1.mlir', 'correlation_1',
              ['chlo.broadcast_divide', 'mhlo.reduce'], False, 3),
    Benchmark('benchmarks/atax.mlir', 'atax',
              ['mhlo.dot', 'chlo.broadcast_add', 'chlo.broadcast_subtract'], True, 3),
    Benchmark('benchmarks/3mm.mlir', '3mm', ['mhlo.dot'], False, 3),
    Benchmark('benchmarks/mvt.mlir', 'mvt',
              ['mhlo.dot', 'chlo.broadcast_add'], False, 3),
    Benchmark('benchmarks/bicg.mlir', 'bicg',
              ['mhlo.dot', 'chlo.broadcast_subtract'], True, 3),
    Benchmark('benchmarks/2mm.mlir', '2mm',
              ['mhlo.dot', 'chlo.broadcast_multiply', 'chlo.broadcast_add'], False, 3),
    Benchmark('benchmarks/gemm.mlir', 'gemm',
              ['chlo.broadcast_add', 'mhlo.dot', 'chlo.broadcast_multiply'], True, 3),
    Benchmark('benchmarks/gesummv.mlir', 'gesummv',
              ['chlo.broadcast_add', 'mhlo.dot', 'chlo.broadcast_multiply'], True, 3),
    Benchmark('benchmarks/syrk.mlir', 'syrk',
              ['mhlo.transpose', 'mhlo.dot', 'chlo.broadcast_multiply', 'mhlo.add', 'mhlo.select'], True, 5),
]

timeout = 300
script_dir = os.path.dirname(os.path.realpath(__file__))
cpu_count = multiprocessing.cpu_count()
program = os.path.join(script_dir, '../build/bin/synthesizer')

def run_program(x):
    start = time.time()
    print(' '.join(x))
    p = subprocess.run(x, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    end = time.time()
    return p.stdout.decode('utf-8'), end-start, p.returncode


def run_benchmark(benchmark, prune_equivalent_candidates, ops, distribute, max_num_ops):
    print('Running benchmark: ' + benchmark.file)

    filename = os.path.join(script_dir, '../' + benchmark.file)
    args = ['--num-threads=%d' % cpu_count,
            '--max-num-ops=%d' % max_num_ops,
            '--print-stats']
    if prune_equivalent_candidates:
        args += ['--ignore-equivalent-candidates']

    if ops == 'ground_truth':
        args += ['--ops=' + ','.join(benchmark.ops)]
    elif ops == 'heuristic':
        args += ['--guide']

    if distribute:
        args += ['--distribute']

    # Run
    out, synth_time, returncode = run_program(
        ['timeout', str(timeout)] + [program, filename] + args)

    # Record stats
    stats = {}
    stats['benchmark_file'] = benchmark.file.split('.')[0]
    stats['benchmark'] = benchmark.name
    stats['prune_equivalent_candidates'] = prune_equivalent_candidates
    stats['operations'] = ops
    stats['distribute'] = distribute
    stats['cmd'] = ' '.join([program, filename] + args)

    stats['status'] = returncode

    if returncode == 0:
        statsStr = out.split('JSON: ')[1].split('\n')[0]
        stats.update(json.loads(statsStr))

        stats['synth_time'] = synth_time
    else:
        print('Synthesis failed')

        # Timeout
        if returncode == 124:
            stats['synth_time'] = synth_time

    return stats


def run_benchmarks_all(benchmarks, prune_eq_configs=[True, False],
              ops_configs=['ground_truth', 'heuristic', 'all'],
              distribute_configs=[True, False]):

    configs = []
    for benchmark in benchmarks:
        for prune_equivalent_candidates in prune_eq_configs:
            for ops in ops_configs:
                for distribute in distribute_configs:
                    configs.append((benchmark, prune_equivalent_candidates, ops, distribute, 6))

    stats_all = []
    for config in tqdm.tqdm(configs):
        stats = run_benchmark(*config)
        print(stats)

        stats_all.append(stats)

    with open('/tmp/stats.json', 'w') as f:
        json.dump(stats_all, f, indent=2)


def run_benchmarks_best(benchmarks):
    stats_all = []
    for benchmark in tqdm.tqdm(benchmarks):
        stats = run_benchmark(benchmark=benchmark,
                              prune_equivalent_candidates=True,
                              ops='ground_truth',
                              distribute=benchmark.distribute,
                              max_num_ops=benchmark.max_num_ops)
        print(stats)

        stats_all.append(stats)

    with open('/tmp/stats.json', 'w') as f:
        json.dump(stats_all, f, indent=2)


def plot_results():
    # Convert to csv.
    with open('/tmp/stats.json', 'r') as f:
        stats_all = json.load(f)
        df = pd.DataFrame(stats_all)
    df.to_csv('/tmp/stats.csv', index=False)

    # Call RScript on plotting script.
    subprocess.run(['Rscript', os.path.join(
        script_dir, 'plot.r'), '/tmp/stats.csv', 'plot_white'])


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run benchmarks')
    parser.add_argument('--exp_all', action='store_true',
                        default=False, help='Run all experiments')
    parser.add_argument('--exp_best', action='store_true',
                        default=False, help='Run experiments in their best configuration')
    args = parser.parse_args()

    if args.exp_all:
        run_benchmarks_all(benchmarks)
    elif args.exp_best:
        run_benchmarks_best(benchmarks)

    # Plot results.
    plot_results()


if __name__ == '__main__':
    main()
