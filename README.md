# ClinicalEye

ClinicalEye is a training-free VLM agent for 3D abdominal CT question answering on DeepTumorVQA.

It builds two anatomy resources and uses them to choose what the model looks at:
- an **anatomy graph**, built from automatic segmentation (TotalSegmentator): structures, slice ranges, contacts, tool lesions and liver segments;
- a cited **anatomy knowledge base**, with location priors for structures the tool does not segment.

A reader VLM states in words what it needs to see. A navigator uses the graph and the knowledge base to locate the anatomy,
renders the views, checks them, and returns an explained image set. The agent never receives benchmark metadata (organ,
lesion, question type or subtype); it gets the question and the CT only.

This repository contains code only. The data must be obtained from its providers (see *Data*). We release:
- the question / case **id lists** of our evaluation sets (`splits/`);
- the structure-probe items;
- our anatomy knowledge base (`knowledge/`);
- everything needed to reproduce the results from the original data.

## Layout

```
ctgraph/                 agent package
  paths.py               all file locations (environment variables, see below)
  volume.py graph.py     CT + segmentation loading (canonical RAS frame), anatomy graph
  navi5.py navi2.py      reader VLM + navigator (method), rendering rules
  agent.py               single-call baseline and shared helpers
  render.py views.py hee.py candidates.py index.py reference.py slicepct.py knowledge.py llm.py
experiments/run_agent.py all agent / single-call conditions of the paper
scripts/                 TotalSegmentator, crop offsets, vLLM serving, external baselines, probes, summary statistics
baselines/3dmedagent/    adaptation patches and helpers for the 3DMedAgent baseline
knowledge/               anatomy knowledge base (JSON + readable Markdown)
splits/                  evaluation question / case ids and probe items
llm_api/example.yaml     template for API model keys (copy, then fill in; real configs are git-ignored)
docs/                    project page (static: index.html + assets/; deploy the folder as is)
```

## Setup

```bash
pip install -r requirements.txt                         # Python 3.10+
git clone https://github.com/Schuture/DeepTumorVQA third_party/DeepTumorVQA   # their prompts, tools, metrics
# separate environments (see requirements.txt): vLLM 0.14 for local VLMs, TotalSegmentator 2.18.0
```

### Data

You need the following. None of it is redistributed here.

| What | Where | Used for |
|---|---|---|
| DeepTumorVQA 2.0: `benchmark/test_qa.jsonl`, `benchmark/test_qa.json`, `benchmark/images_2d/whole/`, `tool_cache/` | `tumor-vqa/DeepTumorVQA_2.0` (HF), request access from the DeepTumorVQA authors | questions, official baselines |
| CT volumes, AbdomenAtlas 1.1 Mini (`<BDMAP_id>/ct.nii.gz`) for the cases in `splits/cases_all.json` | `AbdomenAtlas/_AbdomenAtlas1.1Mini` (HF) | agent input (via TotalSegmentator) |
| AbdomenAtlas 3.0 Mini masks (optional) | `AbdomenAtlas/AbdomenAtlas3.0Mini` (HF), `scripts/prepare_gt_masks.py` | structure probes only; never an agent input |

Point the code at your copies with environment variables. The defaults are shown; all are relative to the repository.

```bash
export CE_DTVQA_DATA=data/DeepTumorVQA_2.0     # contains benchmark/ and tool_cache/
export CE_DTVQA_CODE=third_party/DeepTumorVQA
export CE_ATLAS_CT=data/AbdomenAtlas1.1Mini
export CE_GT_MASKS=data/atlas3_masks           # optional
export CE_WORK=work                            # caches, outputs, LLM response cache, TotalSegmentator outputs
```

### Models

| Model | How |
|---|---|
| Qwen3-VL-32B-Instruct, MedGemma-27B-it, InternVL3.5-38B (FP8) | `scripts/serve_vlm.sh <preset> <gpu>` (vLLM, ports 8000 / 8010 / 8020) |
| GLM-5.3-Flash (thinking disabled), Claude Sonnet 5 | API; copy `llm_api/example.yaml` to `llm_api/keys.yaml` and fill in the keys |

All calls go through `ctgraph/llm.py`, which caches every response on disk and logs tokens to `$CE_WORK/runs/ledger.jsonl`.
Re-running a command resumes for free. `CE_BUDGET_USD` caps API spending.

## Reproduction

```bash
# 1. TotalSegmentator (tasks total, liver_vessels, liver_segments) on the cases we use
CUDA_VISIBLE_DEVICES=0 python scripts/run_totalseg.py --ids-file splits/cases_all.json
# 2. crop offset of the AbdomenAtlas 1.1 copy within the benchmark CT (for "slice %" questions; no ground truth used)
python scripts/build_ts_offsets.py --ids-file splits/cases_all.json
# 3. local VLM server (one per GPU), e.g.
scripts/serve_vlm.sh qwen3-vl-32b 0
# 4. our method, the no-navigator variant and the single-call baselines (anatomy graphs are built on first use)
python experiments/run_agent.py --models qwen3-vl-32b-local \
    --conds clinicaleye no_navigator single_call single_call_no_graph --qids-file splits/lite903.json --workers 4
# 5. DeepTumorVQA's official baselines: direct answer and tool agent (use --adapter prompt for MedGemma)
python scripts/dtvqa_direct.py --model qwen3-vl-32b-local --qids-file splits/lite903.json
python scripts/dtvqa_agent.py  --model qwen3-vl-32b-local --qids-file splits/lite903.json --mode predicted
# 6. tables with case-clustered bootstrap intervals and paired McNemar tests
python scripts/summarize.py --qids-file splits/lite903.json --models qwen3-vl-32b-local \
    --conds direct bench_agent single_call_no_graph single_call no_navigator clinicaleye \
    --pairs clinicaleye:single_call single_call:single_call_no_graph clinicaleye:no_navigator
```

### Conditions (`experiments/run_agent.py --conds`)

| Condition | Paper |
|---|---|
| `clinicaleye` | full method: reader VLM + navigator, anatomy graph + anatomy knowledge |
| `no_navigator` | the reader calls the same rendering rules itself |
| `single_call` / `single_call_no_graph` | one call: organ inferred from the question, overview images, with / without the anatomy-graph summary |
| `clinicaleye_no_overlay`, `no_navigator_no_overlay` | views without drawn outlines (overlay ablation) |
| `no_navigator_memory`, `no_reader_memory` | memory ablations |
| `grounded_navigator`, `render_cap20` | navigator variants |
| `clinicaleye_sample2`, `no_navigator_sample2` | second independent samples (sampling noise) |
| `*_masked_organ` | missing-tool simulation: the question's organ is removed from the segmentation (run on `splits/mask540.json`, `splits/mask_ext363.json`) |

### Splits

| File | Contents |
|---|---|
| `lite903.json` | main evaluation set: 903 questions, 42 subtypes, stratified sample of the DeepTumorVQA test set |
| `lite1k.json` | lite903 plus 123 further questions |
| `mask540.json`, `mask_ext363.json` | questions of the organ-masking experiment (pancreas / kidney; liver / spleen / colon) |
| `pancsub462.json` | questions that need structures the tool does not segment |
| `probe_*_items.json` | structure-probe items (in-plane, out-of-plane, out-of-plane with masked organs) |
| `cases_all.json` | every CT used |

The temperature is 1.0 for our agent and single-call conditions (one sample per condition, `llm_sample` fixed). The
official baselines use temperature 0.

### Structure probes

The probes use ground-truth masks only to build and score the items.

```bash
python scripts/structure_probe.py run qwen3-vl-32b-local 4     # in-plane: which candidate point lies on the structure
python scripts/zprobe.py run qwen3-vl-32b-local 4               # out-of-plane: direction / distance of a structure not in the slice
python scripts/structure_probe.py score; python scripts/zprobe.py score
```

The released item files (`splits/probe_*`) are what we used. To use them, copy them to
`$CE_WORK/runs/structure_probe/items.json` and `$CE_WORK/runs/zprobe/items.json`. Alternatively, rebuild the items with the
`build` sub-commands, which are deterministic.

### 3DMedAgent

See `baselines/3dmedagent/README.md`.
