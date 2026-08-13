import json
for notebook in ['phase_1_w2v.ipynb', 'phase_1_sbert.ipynb']:
    with open('e:\\Backup Laptop 07-08-26\\Develop\\riset-phase-1\\' + notebook, 'r', encoding='utf-8') as f:
        d=json.load(f)
    for c in d['cells']:
        if c['cell_type'] == 'code' and 'summary_data = {' in ''.join(c['source']):
            for i, line in enumerate(c['source']):
                if '"Metric": ["NDCG@5"' in line:
                    c['source'][i] = '    "Metric": ["NDCG@5", "NDCG@10", "MAP", "Spearman rho", "Spearman p-value", "Kendall tau", "Kendall tau p-value"],\n'
                if 'metrics["Spearman"],' in line:
                    c['source'][i] = '        metrics["Spearman"],\n        metrics["Spearman_pvalue"],\n'
                if 'metrics["KendallTau"]' in line:
                    c['source'][i] = '        metrics["KendallTau"],\n        metrics["KendallTau_pvalue"]\n'
    with open('e:\\Backup Laptop 07-08-26\\Develop\\riset-phase-1\\' + notebook, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1)
