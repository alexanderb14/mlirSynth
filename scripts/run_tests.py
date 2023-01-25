import argparse
import json
import multiprocessing
import os
import subprocess
import time
import pandas as pd

tests = [
    ('benchmarks/doitgen.mlir', 'doitgen', ['mhlo.dot_general']),
    ('test/correlation_1.mlir', 'correlation_1', ['chlo.broadcast_divide', 'mhlo.reduce']),
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

# Get script directory
script_dir = os.path.dirname(os.path.realpath(__file__))
timeout = 300

# Run program x and get output as string
def run_program(x):
    start = time.time()
    print(' '.join(x))
    p = subprocess.run(x, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    end = time.time()
    return p.stdout.decode('utf-8'), end-start, p.returncode

def run_tests(tests):
    cpu_count = multiprocessing.cpu_count()
    print('Running experiments on %d cores' % cpu_count)

    program = os.path.join(script_dir, '../build/bin/synthesizer')

    stats_all = []
    for test in tests:
        test_file, test_name, allowed_ops = test
        print('Running test: ' + test_file)

        test = os.path.join(script_dir, '../' + test_file)

        for prune_equivalent_candidates in [True, False]:
            for ops in ['ground_truth', 'heuristic', 'all']:
                for distribute in [True, False]:
                    args = ['--num-threads=%d' % cpu_count,
                            '--max-num-ops=6']
                    if prune_equivalent_candidates:
                        args += ['--ignore-equivalent-candidates']

                    if ops == 'ground_truth':
                        args += ['--ops=' + ','.join(allowed_ops)]
                    elif ops == 'heuristic':
                        args += ['--guide']

                    if distribute:
                        args += ['--distribute']

                    try:
                        # Run synthesis
                        out, synth_time, returncode = run_program([program, test] + args)
                        if returncode != 0:
                            raise RuntimeError('Synthesis failed')

                        # Parse stats
                        statsStr = out.split('JSON: ')[1].split('\n')[0]
                        stats = json.loads(statsStr)
                        stats['synth_time'] = synth_time
                    except (subprocess.TimeoutExpired, RuntimeError) as e:
                        stats = {}
                        stats['synth_time'] = timeout

                    stats['test_file'] = test_file.split('.')[0]
                    stats['benchmark'] = test_name
                    stats['prune_equivalent_candidates'] = prune_equivalent_candidates
                    stats['operations'] = ops
                    stats['distribute'] = distribute
                    stats['cmd'] = ' '.join([program, test] + args)

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
    subprocess.run(['Rscript', os.path.join(script_dir, 'plot.r'), '/tmp/stats.csv'])


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run tests')
    parser.add_argument('--exp', action='store_true', default=False, help='Run experiments')
    args = parser.parse_args()

    if args.exp:
        run_tests(tests)
    
    # Plot results.
    plot_results()


if __name__ == '__main__':
    main()
