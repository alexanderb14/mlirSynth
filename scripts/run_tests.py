import argparse
import json
import os
import subprocess
import time
import pandas as pd
import plotnine as p9

tests = [
    ('correlation_1.mlir', ['chlo.broadcast_divide', 'mhlo.reduce']),
    ('correlation_3.mlir', ['chlo.broadcast_subtract', 'chlo.broadcast_multiply', 'chlo.broadcast_divide']),
    ('atax.mlir', ['mhlo.dot']),
    ('2mm.mlir', ['mhlo.dot', 'chlo.broadcast_multiply', 'chlo.broadcast_add']),
    ('3mm.mlir', ['mhlo.dot']),
    ('mvt_1.mlir', ['mhlo.dot', 'chlo.broadcast_add']),
    ('mvt_2.mlir', ['mhlo.dot', 'chlo.broadcast_add']),
    ('bicg_1.mlir', ['mhlo.dot']),
    ('bicg_2.mlir', ['mhlo.dot']),
    ('gemm.mlir', ['chlo.broadcast_add', 'mhlo.dot', 'chlo.broadcast_multiply']),
    ('gesummv.mlir', ['chlo.broadcast_add', 'mhlo.dot', 'chlo.broadcast_multiply']),
]

# Get script directory
script_dir = os.path.dirname(os.path.realpath(__file__))
timeout = 600

# Run program x and get output as string
def run_program(x):
    start = time.time()
    print(' '.join(x))
    p = subprocess.run(x, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    end = time.time()
    return p.stdout.decode('utf-8'), end-start, p.returncode

def run_tests(tests):
    program = os.path.join(script_dir, '../build/bin/synthesizer')

    stats_all = []
    for test in tests:
        test_file, allowed_ops = test
        print('Running test: ' + test_file)

        test = os.path.join(script_dir, '../test/' + test_file)

        for ignore_equivalent_candidates in [True]:
            for guides in [True]:
                args = ['--num-threads=32', '--max-num-ops=6']
                if ignore_equivalent_candidates:
                    args += ['--ignore-equivalent-candidates']
                if guides:
                    args += ['--ops=' + ','.join(allowed_ops)]

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

                stats['testFile'] = test_file.split('.')[0]
                stats['ignoreEquivalentCandidates'] = ignore_equivalent_candidates
                stats['guides'] = guides

                print(stats)
                stats_all.append(stats)

                with open('/tmp/stats.json', 'w') as f:
                    json.dump(stats_all, f, indent=2)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run tests')
    parser.add_argument('--exp', action='store_true', default=False, help='Run experiments')
    args = parser.parse_args()

    if args.exp:
        run_tests(tests)
    
    # Plot results.
    with open('/tmp/stats.json', 'r') as f:
        stats_all = json.load(f)
        df = pd.DataFrame(stats_all)
    df.to_csv('/tmp/stats.csv', index=False)
    
    df = pd.read_csv('/tmp/stats.csv')
    plot = (p9.ggplot(df[df['ignoreEquivalentCandidates']==True],
            p9.aes(x='testFile', y='synth_time', fill='guides'))
    + p9.geom_col(stat="identity", width=.5, position = "dodge")
    + p9.scale_y_sqrt()
    + p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1)))
    plot.save('/tmp/plot.pdf', width=10, height=5)


if __name__ == '__main__':
    main()
