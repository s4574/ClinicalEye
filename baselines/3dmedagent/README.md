# 3DMedAgent baseline

We ran the released 3DMedAgent pipeline (arXiv 2602.18064) on our question sets with the same base models as our method.
This folder has everything needed to reproduce that, but not their code.

| Component | Source | Version |
|---|---|---|
| 3DMedAgent | `github.com/jinlab-imvr/3DMedAgent` | commit in `3dmedagent_base_commit.txt` |
| MONAI VISTA | `github.com/Project-MONAI/VISTA` | `d4a8fe0dbf5cb4b76c531fccca8e29c1e6f6ee45` |
| CT-CLIP | `github.com/ibrahimethemhamamci/CT-CLIP` | `a2a155c601987820433c01db69b64d701d3d229d` |
| VISTA3D weights | HF `nvidia/NV-Segment-CT`, `vista3d_pretrained_model/model_monai1.3.pt` | |
| CT-CLIP weights | HF `ibrahimhamamci/CT-RATE` (gated), `models/CT-CLIP-Related/CT-CLIP_v2.pt` | |
| CTs | DeepTumorVQA 2.0, `benchmark/ct/<id>/ct.nii.gz` | |

The segmentation backend is VISTA3D, which is the paper's setting.

## Files

- `3dmedagent_adaptations.diff`: every change to their repository. Apply it with `git apply` on the base commit, then set
  `bundle_root` in `SegAgent/VISTA3d/configs/infer.yaml` to your VISTA3D bundle directory. The changes:
  - hard-coded paths become environment variables;
  - the segmentation back-ends are imported lazily;
  - the DeepTumorVQA branch of `Structural_Report/main.py` is restored;
  - `$FINAL_TEST_EXTRA_BODY` is passed to the model (needed for GLM-5.3-Flash: `{"thinking": {"type": "disabled"}}`);
  - token accounting.

  Prompts, tools, policies, retries and decoding settings are unchanged.
- `ct_tree_fairness_infer_qtype.diff`: the optional variant without metadata. It is gated by `EXT3DMA_INFER_QTYPE=1` and off
  by default. With it on, the benchmark's question type is not put into the prompts; the same model infers it from the question
  (one extra text call).
- `build_vqa_csv.py`: writes their VQA table for one of our splits. `--mask-metadata` blanks the `organ` / `lesion` columns.
  Their tools use those columns as a fallback target, and our agents never receive that metadata.
- `score_predictions.py`: per-question scoring of the saved records. It joins on `qid`, so results pair with ours.

## Steps

1. Segment with VISTA3D (`segmentation.py`).
2. Build the structured reports (`Structural_Report/main.py`).
3. Run the four CT-CLIP stages (`CT_Clip/*.py`).
4. Run the reasoning stage once per question type:

   ```
   python Final_Test/GPT_memory_t1s.py --question-type "<type>" --max-per-subtype 0 \
       --normalizer rule --memory-mode facts --include-runtime-tools --tool-selector gpt --include-t1s \
       --t1s-max-iters 5 --model <served name> --vision-model <served name> --skip-existing
   ```

   Loop over the four types `recognition`, `measurement`, `visual reasoning` and `medical reasoning`. Without
   `--question-type`, the script falls back to four hard-coded subtypes.

   For the variant without metadata, build the table with `--mask-metadata` and set `EXT3DMA_INFER_QTYPE=1`.
5. Score with `python score_predictions.py --vqa-csv <table> --save-dir <records> --out-dir <out>`.

Local models are served with `../../scripts/serve_vlm.sh`, which gives the same weights and settings as our runs.
