import json
import os
import subprocess
import time

# Get script directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Run program x and get output as string
def run_program(x):
    start = time.time()
    p = subprocess.Popen(x, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    end = time.time()
    return out.decode("utf-8"), end - start, p.returncode

program = os.path.join(script_dir, '../build/bin/synthesizer')
args_add = ['--num-threads=32', '--ignore-equivalent-candidates']

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

stats_all = []
for test in tests:
    test_file, test_allowed_ops = test
    print('Running test: ' + test_file)

    test = os.path.join(script_dir, '../test/' + test_file)

    for allowed_ops in [test_allowed_ops, []]:
        allowed_ops_arg = '--ops=' + ','.join(allowed_ops)

        # Run synthesis
        args = args_add + [allowed_ops_arg]
        out, synth_time, returncode = run_program([program, test] + args)
    
        # Parse stats
        statsStr = out.split('JSON: ')[1].split('\n')[0]
        stats = json.loads(statsStr)
    
        stats['test_file'] = test_file
        stats['args'] = ','.join(args)
        stats['synth_time'] = synth_time
    
        print(stats)
        stats_all.append(stats)
    
        with open('/tmp/stats.json', 'w') as f:
            json.dump(stats_all, f, indent=2)
