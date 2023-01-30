import tqdm

import argparse
import json
import multiprocessing
import os
import subprocess
import time
import pandas as pd

# fmt: off
tests = [
    ('benchmarks/doitgen.mlir', 'doitgen', ['mhlo.dot_general']),
    ('testfiles/correlation_1.mlir', 'correlation_1', ['chlo.broadcast_divide', 'mhlo.reduce']),
#    ('test/correlation_3.mlir', 'correlation_3', ['chlo.broadcast_subtract', 'chlo.broadcast_multiply', 'chlo.broadcast_divide']),
    ('benchmarks/atax.mlir', 'atax', ['mhlo.dot','chlo.broadcast_add','chlo.broadcast_subtract']),
    ('benchmarks/3mm.mlir', '3mm', ['mhlo.dot']),
    ('benchmarks/mvt.mlir', 'mvt', ['mhlo.dot', 'chlo.broadcast_add']),
    ('benchmarks/bicg.mlir', 'bicg', ['mhlo.dot', 'chlo.broadcast_subtract']),
    ('benchmarks/2mm.mlir', '2mm', ['mhlo.dot', 'chlo.broadcast_multiply', 'chlo.broadcast_add']),
    ('benchmarks/gemm.mlir', 'gemm', ['chlo.broadcast_add', 'mhlo.dot', 'chlo.broadcast_multiply']),
#    ('benchmarks/gemver.mlir', 'gemver', []),
    ('benchmarks/gesummv.mlir', 'gesummv', ['chlo.broadcast_add', 'mhlo.dot', 'chlo.broadcast_multiply']),
]
# fmt: on

# Get script directory
script_dir = os.path.dirname(os.path.realpath(__file__))
timeout = 300


def run_program(x):
    start = time.time()
    print(' '.join(x))
    p = subprocess.run(x, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    end = time.time()
    return p.stdout.decode('utf-8'), end-start, p.returncode


def run_tests(tests):
    cpu_count = multiprocessing.cpu_count()
    print('Running experiments on %d cores' % cpu_count)

    program = os.path.join(script_dir, '../build/bin/synthesizer')

    stats_all = []
    for test in tqdm.tqdm(tests):
        test_file, test_name, allowed_ops = test
        print('Running test: ' + test_file)

        test = os.path.join(script_dir, '../' + test_file)

        for prune_equivalent_candidates in [True, False]:
            for ops in ['ground_truth', 'heuristic', 'all']:
                for distribute in [True, False]:
                    args = ['--num-threads=%d' % cpu_count,
                            '--print-stats',
                            '--max-num-ops=6']
                    if prune_equivalent_candidates:
                        args += ['--ignore-equivalent-candidates']

                    if ops == 'ground_truth':
                        args += ['--ops=' + ','.join(allowed_ops)]
                    elif ops == 'heuristic':
                        args += ['--guide']

                    if distribute:
                        args += ['--distribute']

                    # Run
                    out, synth_time, returncode = run_program(
                        ['timeout', str(timeout)] + [program, test] + args)

                    # Record stats
                    stats = {}
                    stats['test_file'] = test_file.split('.')[0]
                    stats['benchmark'] = test_name
                    stats['prune_equivalent_candidates'] = prune_equivalent_candidates
                    stats['operations'] = ops
                    stats['distribute'] = distribute
                    stats['cmd'] = ' '.join([program, test] + args)

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
    parser = argparse.ArgumentParser(description='Run tests')
    parser.add_argument('--exp', action='store_true',
                        default=False, help='Run experiments')
    args = parser.parse_args()

    if args.exp:
        run_tests(tests)

    # Plot results.
    plot_results()


if __name__ == '__main__':
    main()
