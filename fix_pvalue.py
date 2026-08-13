import json
import re
for notebook in ['phase_1_tfidf.ipynb', 'phase_1_w2v.ipynb', 'phase_1_sbert.ipynb']:
    with open('e:\\Backup Laptop 07-08-26\\Develop\\riset-phase-1\\' + notebook, 'r', encoding='utf-8') as f:
        d=json.load(f)
    for c in d['cells']:
        if c['cell_type'] == 'code':
            for i, line in enumerate(c['source']):
                if '"Spearman_pvalue": round(float(spearman_p), 6)' in line:
                    c['source'][i] = line.replace('round(float(spearman_p), 6)', 'float(spearman_p)')
                if '"KendallTau_pvalue": round(float(kendall_p), 6)' in line:
                    c['source'][i] = line.replace('round(float(kendall_p), 6)', 'float(kendall_p)')
    with open('e:\\Backup Laptop 07-08-26\\Develop\\riset-phase-1\\' + notebook, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1)
