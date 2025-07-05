from replace_distances import replace
from dep.dep import dep
from el.el import el
from mt.mt import mt
from pos.pos import pos
import pandas as pd

methods = ['ant']
tasks = ['mt', 'dep', 'el', 'pos']

# Change this to 'ALL' to use all sub-task relevant features (incl. distances)
# or 'URIEL' to use only the language distances
FEATURE_SET = 'URIEL'

if FEATURE_SET == 'ALL':
    features_dep = ['Word overlap','Transfer lang dataset size',
                    'Target lang dataset size', 'Transfer over target size ratio', 'Transfer lang TTR',
                    'Target lang TTR', 'Transfer target TTR distance','GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']
    features_el = ['Entity overlap','Transfer lang dataset size','Target lang dataset size',
                    'Transfer over target size ratio','GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']
    features_mt = [
            'Overlap word-level', 'Overlap subword-level', 'Transfer lang dataset size',
            'Target lang dataset size', 'Transfer over target size ratio', 'Transfer lang TTR',
            'Target lang TTR', 'Transfer target TTR distance', 'GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC'
        ]
    features_pos = ['Overlap word-level','Transfer lang dataset size','Target lang dataset size','Transfer over target size ratio',
                    'Transfer lang TTR','Target lang TTR','Transfer target TTR distance',
                    'GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']
elif FEATURE_SET == 'URIEL':
    features_dep = ['GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']
    features_el = ['GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']
    features_mt = ['GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']
    features_pos = ['GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']
results = {method: {task: [] for task in tasks} for method in methods}
for method in methods:
    results[method]['num_features'] = []

# # baseline
# replace('URIELPlus_Union.csv')
# dep_ndcg = dep('dep/dep_selected.csv', features_dep)
# el_ndcg = el('el/el_selected.csv', features_el)
# mt_ndcg = mt('mt/mt_selected.csv', features_mt)
# pos_ndcg = pos('pos/pos_selected.csv', features_pos)
# results['baseline'] = {
#     'dep': [dep_ndcg],
#     'el': [el_ndcg],
#     'mt': [mt_ndcg],
#     'pos': [pos_ndcg]
# }
# # Save baseline results to CSV
# baseline_df = pd.DataFrame(results['baseline'])
# baseline_df.to_csv('results/baseline_results.csv', index=False)

for method in methods:
    for i in range(100, 701, 100):
        replace(f'selection_result/imputed_{method}_{i}.csv')
        dep_ndcg = dep('dep/dep_selected.csv', features_dep)
        el_ndcg = el('el/el_selected.csv', features_el)
        mt_ndcg = mt('mt/mt_selected.csv', features_mt)
        pos_ndcg = pos('pos/pos_selected.csv', features_pos)
        results[method]['dep'].append(dep_ndcg)
        results[method]['el'].append(el_ndcg)
        results[method]['mt'].append(mt_ndcg)
        results[method]['pos'].append(pos_ndcg)
        results[method]['num_features'].append(i)
        print(f'Finished {method} with {i} features')
    results_df = pd.DataFrame(results[method])
    results_df.set_index('num_features', inplace=True)
    results_df.to_csv(f'results/{method}_results.csv')