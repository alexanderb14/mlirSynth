import json
import pandas as pd


def main():
    with open('/tmp/stats.json', 'r') as f:
        stats_all = json.load(f)

    df_rows = []
    for stats in stats_all:
        if 'processingStatusCounts' not in stats:
            continue

        num_failed_static_checks = sum(stats['processingStatusCounts'][k] for k in [
                                       'reject_isNotVerifiable', 'reject_isNotResultTypeInferrable'])

        # Add row to dataframe with benchmark name and stats.
        df_rows.append({'benchmark': stats['benchmark'],
                        'enumerated': stats['numEnumerated'],
                        'evaluated': stats['numExecuted'],
                        'failed_static_checks': num_failed_static_checks,
                        'failed_equivalence_check': stats['processingStatusCounts']['reject_hashNotUnique'],
                        'time': stats['synth_time']})

    print(df_rows)
    df = pd.DataFrame(df_rows)

    tex = df.to_latex(index=False, escape=False, column_format='lrrrrr')
    tex = tex.replace('_', '\_')
    print(tex)
    

if __name__ == '__main__':
    main()
