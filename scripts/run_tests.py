import argparse
import collections
import json
import multiprocessing
import os
import shutil
import subprocess
import time

import tqdm
import pandas as pd


Benchmark = collections.namedtuple(
    'Benchmark', ['file', 'name', 'ops', 'distribute', 'max_num_ops'])

benchmarks_hlo = [
    Benchmark('benchmarks/doitgen.mlir', 'doitgen',
              ['stablehlo.dot_general'], False, 3),
    Benchmark('benchmarks/atax.mlir', 'atax',
              ['stablehlo.dot', 'chlo.broadcast_add', 'chlo.broadcast_subtract'], True, 3),
    Benchmark('benchmarks/3mm.mlir', '3mm', ['stablehlo.dot'], False, 3),
    Benchmark('benchmarks/mvt.mlir', 'mvt',
              ['stablehlo.dot', 'chlo.broadcast_add'], False, 3),
    Benchmark('benchmarks/bicg.mlir', 'bicg',
              ['stablehlo.dot', 'chlo.broadcast_subtract'], True, 3),
    Benchmark('benchmarks/2mm.mlir', '2mm',
              ['stablehlo.dot', 'chlo.broadcast_multiply', 'chlo.broadcast_add'], False, 3),
    Benchmark('benchmarks/gemm.mlir', 'gemm',
              ['chlo.broadcast_add', 'stablehlo.dot', 'chlo.broadcast_multiply'], True, 3),
    Benchmark('benchmarks/gesummv.mlir', 'gesummv',
              ['chlo.broadcast_add', 'stablehlo.dot', 'chlo.broadcast_multiply'], True, 3),
    Benchmark('benchmarks/gemver.mlir', 'gemver',
              ['stablehlo.dot', 'chlo.broadcast_add', 'chlo.broadcast_multiply'], False, 3),
    Benchmark('benchmarks/covariance.mlir', 'covariance',
              ['stablehlo.dot_general', 'stablehlo.transpose', 'chlo.broadcast_divide', 'chlo.broadcast_subtract', 'stablehlo.reduce'], False, 3),
    Benchmark('benchmarks/correlation.mlir', 'correlation',
              ['stablehlo.dot_general', 'chlo.broadcast_divide', 'chlo.broadcast_subtract'], False, 3),
    Benchmark('benchmarks/symm.mlir', 'symm',
              ['stablehlo.transpose', 'stablehlo.dot', 'chlo.broadcast_multiply', 'stablehlo.add', 'stablehlo.select'], False, 5),
    Benchmark('benchmarks/syrk.mlir', 'syrk',
              ['stablehlo.transpose', 'stablehlo.dot', 'chlo.broadcast_multiply', 'stablehlo.add', 'stablehlo.select'], True, 5),
    Benchmark('benchmarks/syr2k.mlir', 'syr2k',
              ['stablehlo.transpose', 'stablehlo.dot', 'chlo.broadcast_multiply', 'stablehlo.add', 'stablehlo.select'], True, 5),
    Benchmark('benchmarks/trmm.mlir', 'trmm',
              ['stablehlo.dot'], False, 3),
]

benchmarks_linalg = [
    Benchmark('benchmarks/2mm.mlir', '2mm',
              ['linalg.matmul'], True, 1),
    Benchmark('benchmarks/3mm.mlir', '3mm',
              ['linalg.matvec'], True, 1),
    Benchmark('benchmarks/bicg.mlir', 'bicg',
              ['linalg.matvec'], True, 1),
    # Benchmark('benchmarks/from_mlt/gemm.mlir', 'gemm',
    #           ['linalg.matmul', 'linalg.matvec'], True, 1),
    # Benchmark('benchmarks/gemver.mlir', 'gemver',
    #           ['linalg.matmul', 'linalg.matvec'], True, 1),
    Benchmark('benchmarks/gesummv.mlir', 'gesummv',
              ['linalg.matvec'], True, 1),
    Benchmark('benchmarks/mvt.mlir', 'mvt',
              ['linalg.matvec'], True, 1),
]

TIMEOUT = 60 * 60
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RES_DIR = '/tmp/exp_results'
CPU_COUNT = multiprocessing.cpu_count()
SYNTHESIZER_PROGRAM = os.path.join(SCRIPT_DIR, '../build/bin/synthesizer')
FILECHECK_PROGRAM = os.path.join(SCRIPT_DIR, '../mlir-hlo/llvm-build/bin/FileCheck')


def run_program(cmd, stdin=None):
    start = time.time()
    print(' '.join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, input=stdin,
                       stderr=subprocess.PIPE)
    end = time.time()
    return p.stdout.decode('utf-8'), end-start, p.returncode


def run_benchmark(dialect, benchmark, prune_equivalent_candidates, ops, distribute, max_num_ops):
    print('Running benchmark: ' + benchmark.file)

    filename = os.path.join(SCRIPT_DIR, '../' + benchmark.file)
    args = ['--target-dialect=%s' % dialect,
            '--num-threads=%d' % CPU_COUNT,
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
    synth_out, synth_time, synth_returncode = run_program(
        ['timeout', str(TIMEOUT)] + [SYNTHESIZER_PROGRAM, filename] + args)

    with open(os.path.join(RES_DIR, benchmark.name + '.stdout'), 'w') as f:
        f.write(synth_out)

    # Check
    # check_out, check_time, check_returncode = run_program(
    #     [FILECHECK_PROGRAM, filename], stdin=synth_out.encode('utf-8'))
    check_returncode = 0

    # Record stats
    stats = {}
    stats['benchmark_file'] = benchmark.file.split('.')[0]
    stats['benchmark'] = benchmark.name
    stats['prune_equivalent_candidates'] = prune_equivalent_candidates
    stats['operations'] = ops
    stats['distribute'] = distribute
    stats['cmd'] = ' '.join([SYNTHESIZER_PROGRAM, filename] + args)

    stats['status'] = synth_returncode

    if synth_returncode == 0 and check_returncode == 0:
        print('\033[1;42mSynthesis success\033[0m')

        statsStr = synth_out.split('JSON: ')[1].split('\n')[0]
        stats.update(json.loads(statsStr))

        stats['synth_time'] = synth_time
    else:
        print('\033[1;41mSynthesis failed\033[0m ' +
              'synth_returncode: %d, check_returncode: %d' %
              (synth_returncode, check_returncode))

        # Timeout
        if synth_returncode == 124:
            stats['synth_time'] = synth_time

    return stats


def run_benchmarks_all(benchmarks, dialect, prune_eq_configs=[True, False],
                       ops_configs=['ground_truth', 'heuristic', 'all'],
                       distribute_configs=[True, False]):

    configs = []
    for benchmark in benchmarks:
        for prune_equivalent_candidates in prune_eq_configs:
            for ops in ops_configs:
                for distribute in distribute_configs:
                    configs.append(
                        (dialect, benchmark, prune_equivalent_candidates, ops, distribute, 6))

    stats_all = []
    for config in tqdm.tqdm(configs):
        stats = run_benchmark(*config)

        print(stats)
        stats_all.append(stats)
        with open('/tmp/stats.json', 'w') as f:
            json.dump(stats_all, f, indent=2)


def run_benchmarks_best(benchmarks, dialect):
    stats_all = []
    for benchmark in tqdm.tqdm(benchmarks):
        stats = run_benchmark(benchmark=benchmark,
                              dialect=dialect,
                              prune_equivalent_candidates=True,
                              ops="ground_truth",
                              distribute=benchmark.distribute,
                              max_num_ops=benchmark.max_num_ops)
        print(stats)
        stats_all.append(stats)
        with open('/tmp/stats.json', 'w') as f:
            json.dump(stats_all, f, indent=2)


def run_benchmarks_heuristic(benchmarks, dialect):
    stats_all = []
    for benchmark in tqdm.tqdm(benchmarks):
        stats = run_benchmark(benchmark=benchmark,
                              dialect=dialect,
                              prune_equivalent_candidates=True,
                              ops="heuristic",
                              distribute=benchmark.distribute,
                              max_num_ops=5)
        print(stats)
        stats_all.append(stats)
        with open('/tmp/stats.json', 'w') as f:
            json.dump(stats_all, f, indent=2)


def run_benchmarks_naive(benchmarks, dialect):
    stats_all = []
    for benchmark in tqdm.tqdm(benchmarks):
        stats = run_benchmark(benchmark=benchmark,
                              dialect=dialect,
                              prune_equivalent_candidates=True,
                              ops="all",
                              distribute=benchmark.distribute,
                              max_num_ops=5)
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
        SCRIPT_DIR, 'plot.r'), '/tmp/stats.csv', 'plot_white'])


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run benchmarks')
    parser.add_argument('--exp_all', action='store_true',
                        default=False, help='Run all experiments')
    parser.add_argument('--exp_best', action='store_true',
                        default=False, help='Run experiments in their best configuration')
    parser.add_argument('--exp_heuristic', action='store_true',
                        default=False, help='Run experiments with heuristic')
    parser.add_argument('--exp_naive_heuristic', action='store_true',
                        default=False, help='Run experiments with naive and heuristic')
    parser.add_argument('--dialect', type=str, help='Dialect to use')
    parser.add_argument('--benchmarks', type=str, nargs='+', help='Benchmarks to run')
    args = parser.parse_args()

    if os.path.exists(RES_DIR):
        shutil.rmtree(RES_DIR)
    os.mkdir(RES_DIR)

    benchmarks = None
    if args.dialect == 'hlo':
        benchmarks = benchmarks_hlo
    elif args.dialect == 'linalg':
        benchmarks = benchmarks_linalg
    else:
        raise Exception('Unknown dialect')

    if args.benchmarks:
        benchmarks = [b for b in benchmarks if b.name in args.benchmarks]

    if args.exp_all:
        run_benchmarks_all(benchmarks, args.dialect)
    elif args.exp_best:
        run_benchmarks_best(benchmarks, args.dialect)
    elif args.exp_heuristic:
        run_benchmarks_heuristic(benchmarks, args.dialect)
    elif args.exp_naive_heuristic:
        run_benchmarks_heuristic(benchmarks, args.dialect)
        run_benchmarks_naive(benchmarks, args.dialect)

    # Plot results.
    plot_results()


if __name__ == '__main__':
    main()
