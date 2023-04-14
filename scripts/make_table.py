import json
import pandas as pd


def main():
    with open('/tmp/stats.json', 'r') as f:
        stats_all = json.load(f)

    df_rows = []


    # Iterate stats_all, sorted by benchmark name.
    for stats in sorted(stats_all, key=lambda x: x['benchmark']):
        if stats['operations'] != 'heuristic':
            continue

        if 'processingStatusCounts' not in stats:
            continue

        num_failed_static_checks = sum(stats['processingStatusCounts'][k] for k in [
                                       'reject_isNotVerifiable', 'reject_isNotResultTypeInferrable'])

        synth_time = stats['synth_time']
        synth_time = round(synth_time, 2)

        # Add row to dataframe with benchmark name and stats.
        df_rows.append({'Benchmark': stats['benchmark'],
                        'Enumerated': stats['numEnumerated'],
                        'Static filtered': num_failed_static_checks,
                        'Evaluated': stats['numExecuted'],
                        'Equivalence filtered': stats['processingStatusCounts']['reject_hashNotUnique'],
                        'Synthesis time (in s)': synth_time})

    print(df_rows)
    df = pd.DataFrame(df_rows)

    tex = df.to_latex(index=False, escape=False, column_format='lrrrrr')
    tex = tex.replace('_', '\_')
    print(tex)
    

if __name__ == '__main__':
    main()
