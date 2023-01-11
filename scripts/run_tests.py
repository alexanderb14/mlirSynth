import json
import os
import subprocess
import time
import pandas as pd

# Get script directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Run program x and get output as string
def run_program(x):
    start = time.time()
    p = subprocess.run(x, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    end = time.time()
    return p.stdout.decode('utf-8'), end-start, p.returncode

def run_tests(tests):
    program = os.path.join(script_dir, '../build/bin/synthesizer')

    stats_all = []
    for test in tests:
        test_file, allowed_ops = test
        print('Running test: ' + test_file)

        test = os.path.join(script_dir, '../test/' + test_file)

        for ignore_equivalent_candidates in [True, False]:
            for restrict_ops in [True, False]:
                args = ['--num-threads=32']
                if ignore_equivalent_candidates:
                    args += ['--ignore-equivalent-candidates']
                if restrict_ops:
                    args += ['--ops=' + ','.join(allowed_ops)]

                try:
                    # Run synthesis
                    out, synth_time, returncode = run_program([program, test] + args)

                    # Parse stats
                    statsStr = out.split('JSON: ')[1].split('\n')[0]
                    stats = json.loads(statsStr)
                    stats['synth_time'] = synth_time
                except subprocess.TimeoutExpired:
                    stats = {}

                stats['test_file'] = test_file
                stats['ignore_equivalent_candidates'] = ignore_equivalent_candidates
                stats['restrict_ops'] = restrict_ops

                print(stats)
                stats_all.append(stats)

                with open('/tmp/stats.json', 'w') as f:
                    json.dump(stats_all, f, indent=2)

tests = [
    ('correlation_1.mlir', ['chlo.broadcast_divide', 'mhlo.reduce']),
    ('atax.mlir', ['mhlo.dot']),
    ('2mm.mlir', ['mhlo.dot', 'mhlo.multiply']),
    ('3mm.mlir', ['mhlo.dot']),
    ('mvt_1.mlir', ['mhlo.dot', 'chlo.broadcast_add']),
    ('mvt_2.mlir', ['mhlo.dot', 'chlo.broadcast_add']),
    ('bicg_1.mlir', ['mhlo.dot']),
    ('bicg_2.mlir', ['mhlo.dot']),
    ('gemm.mlir', ['chlo.broadcast_add', 'mhlo.dot', 'chlo.broadcast_multiply']),
    ('gesummv.mlir', ['chlo.broadcast_add', 'mhlo.dot', 'chlo.broadcast_multiply']),
]
run_tests(tests)

with open('/tmp/stats.json', 'r') as f:
    stats_all = json.load(f)
    df = pd.DataFrame(stats_all)
print(df)
