"""All file locations, configurable through environment variables (defaults: inside the repository).

  CE_WORK          working directory for caches, outputs and the LLM response cache      (default: <repo>/work)
  CE_DTVQA_DATA    DeepTumorVQA 2.0 dataset snapshot (HF `tumor-vqa/DeepTumorVQA_2.0`; needs `benchmark/` and
                   `tool_cache/`)                                                          (default: <repo>/data/DeepTumorVQA_2.0)
  CE_DTVQA_CODE    clone of the DeepTumorVQA code (github.com/Schuture/DeepTumorVQA)       (default: <repo>/third_party/DeepTumorVQA)
  CE_ATLAS_CT      AbdomenAtlas1.1Mini CT volumes (<BDMAP_id>/ct.nii.gz)                   (default: <repo>/data/AbdomenAtlas1.1Mini)
  CE_GT_MASKS      AbdomenAtlas3.0Mini masks, extracted per case (<root>/bench/<BDMAP_id>/segmentations/*.nii.gz); only for
                   the probes and GT-based analyses, never an input of the agent           (default: <repo>/data/atlas3_masks)
  CE_TOTALSEG_OUT  TotalSegmentator outputs (<BDMAP_id>/{total,liver_vessels,liver_segments}.nii.gz)
                                                                                           (default: <CE_WORK>/totalseg)
  CE_LLM_API       directory with API model configs (*.yaml, see llm_api/example.yaml)     (default: <repo>/llm_api)
  CE_LLM_CACHE     LLM response cache                                                      (default: <CE_WORK>/llm_cache)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _p(env: str, default) -> Path:
    return Path(os.environ.get(env, str(default))).expanduser()


WORK = _p("CE_WORK", REPO / "work")
DTVQA_DATA = _p("CE_DTVQA_DATA", REPO / "data/DeepTumorVQA_2.0")
DTVQA_CODE = _p("CE_DTVQA_CODE", REPO / "third_party/DeepTumorVQA")
ATLAS_CT = _p("CE_ATLAS_CT", REPO / "data/AbdomenAtlas1.1Mini")
GT_MASKS = _p("CE_GT_MASKS", REPO / "data/atlas3_masks")
TOTALSEG_OUT = _p("CE_TOTALSEG_OUT", WORK / "totalseg")
LLM_API = _p("CE_LLM_API", REPO / "llm_api")
LLM_CACHE = _p("CE_LLM_CACHE", WORK / "llm_cache")

CACHE = WORK / "cache"                     # case arrays (.npz), anatomical graphs, crop offsets
RUNS = WORK / "runs"                       # episodes, baselines, probes, ledger
TEST_QA = DTVQA_DATA / "benchmark/test_qa.jsonl"
TOOL_CACHE = DTVQA_DATA / "tool_cache"
SPLITS = REPO / "splits"
ANATOMY_KB = REPO / "knowledge/anatomy_kb.json"
DTVQA_SRC = DTVQA_CODE / "src"
