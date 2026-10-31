"""Build notebooks/02_data_exploration.ipynb (exploration + one figure). Execute with:
   jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_exploration.ipynb"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = [
nbf.v4.new_markdown_cell("# Phase 2 — MedQA data and variants\nExploration only (no results claims). Inputs: "
    "`data/raw/medqa_us4/*.jsonl`, `data/processed/medqa_us4_variants/{dev,test}.jsonl` from `scripts/make_variants.py`."),
nbf.v4.new_code_cell("""import sys, json, collections
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd, matplotlib.pyplot as plt
from src.data.medqa import load_split
from src.data.bias import TEMPLATES, FAMILY_OF, BIASMEDQA
splits = {s: load_split(s) for s in ('train', 'dev', 'test')}
pd.DataFrame({s: {'questions': len(v)} for s, v in splits.items()})"""),
nbf.v4.new_markdown_cell("## Answer-letter balance and exam step"),
nbf.v4.new_code_cell("""rows = [{'split': s, 'gold': it['answer_idx'], 'step': it.get('meta_info'), 'words': len(it['question'].split())}
        for s, v in splits.items() for it in v]
df = pd.DataFrame(rows)
display(pd.crosstab(df.split, df.gold, normalize='index').round(3))
display(pd.crosstab(df.split, df.step))
df.groupby('split').words.describe().round(1)"""),
nbf.v4.new_markdown_cell("## Variants (dev): counts per kind and where misleading cues point"),
nbf.v4.new_code_cell("""dev = [json.loads(l) for l in open(ROOT / 'data/processed/medqa_us4_variants/dev.jsonl')]
kinds = collections.Counter(v['variant'].split(':')[0] for v in dev)
print(kinds)
mis = pd.DataFrame([{'bias': v['variant'].split(':')[1], 'family': v['family'], 'target': v['cue_target'],
                     'gold': v['answer_idx'], 'added_words': len(v['question'].split())} for v in dev if v['variant'].startswith('mis:')])
src_words = {it['id']: len(it['question'].split()) for it in splits['dev']}
mis['added_words'] -= [src_words[v['src_id']] for v in dev if v['variant'].startswith('mis:')]
display(pd.crosstab(mis.gold, mis.target))   # diagonal must be 0
mis.groupby(['family', 'bias']).added_words.mean().round(1).to_frame('mean words added')"""),
nbf.v4.new_markdown_cell("## Validation manifest"),
nbf.v4.new_code_cell("""man = json.load(open(ROOT / 'results/data/variants_manifest.json'))
print(json.dumps({s: {k: man['splits'][s][k] for k in ('source_questions', 'variant_rows', 'issues', 'nota_ambiguous_questions')}
                  for s in man['splits']}, indent=1))
print(json.dumps(man['medqa_duplicates'], indent=1)[:600])"""),
nbf.v4.new_markdown_cell("## Figure: question length and cue length"),
nbf.v4.new_code_cell("""fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for s in ('train', 'dev', 'test'):
    ax[0].hist(df[df.split == s].words, bins=40, histtype='step', density=True, label=s)
ax[0].set_xlabel('words per question'); ax[0].set_ylabel('density'); ax[0].legend()
m = mis.groupby('bias').added_words.mean().reindex(list(TEMPLATES))
ax[1].barh(range(len(m)), m.values, color=['#5d8a91' if b in BIASMEDQA else '#0e6a72' for b in m.index])
ax[1].set_yticks(range(len(m))); ax[1].set_yticklabels(m.index, fontsize=8); ax[1].invert_yaxis()
ax[1].set_xlabel('words added by the cue (dev, misleading)')
fig.tight_layout(); fig.savefig(ROOT / 'figures/02_data_overview.png', dpi=150)"""),
]
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks/02_data_exploration.ipynb"
nbf.write(nb, out); print("written", out)
