# Hallucination Detection

LLM-backed claim-level medical hallucination detector for RadClaw output folders.

The implementation follows `hallucination.md`:

- LLM claim extraction for generated/reference Findings and Impression sections
- Stage A RadFact-style LLM support verification
- Explicit CRIMSON-style Stage B matching for Findings with radiology synonym normalization
- LLM Findings-Impression internal consistency checks
- LLM recommendation support checks
- Final LLM taxonomy classification into H1-H10 plus omission events, with Stage B results enforced in post-processing

## Current Evaluation Semantics

- Findings Stage B is implemented explicitly: generated/reference Findings are normalized with a common radiology synonym dictionary, matched by core finding/anatomy, then classified into H1/H2/H3/H4/H5/H6/H7/H9 before final LLM aggregation.
- Image evidence is read from each case `images/` directory and enriched with descriptions from `sample_xxx/image_index.json`. These descriptions are included in prompts, and raster images are uploaded by default for multimodal models.
- `unverifiable` generated claims are not counted as hallucination errors. They are reported separately under `num_unverifiable_generated_claims`, `unverifiable_rate`, and `unverifiable_claim_ids`.

## Usage

Prefer passing the API key through an environment variable:

```bash
export OPENAI_API_KEY="..."
python -m hallucination_detection \
  --target-dir output/mimic/full_radclaw \
  --model YOUR_MODEL_NAME \
  --output-name hallucination_detection.json
```

For an OpenAI-compatible local or hosted endpoint:

```bash
export OPENAI_API_KEY="..."
python -m hallucination_detection \
  --target-dir output/mimic/full_radclaw \
  --model YOUR_MODEL_NAME \
  --base-url http://localhost:8000/v1
```

Each processed case receives `sample_xxx/hallucination_detection.json`. A target-level
`hallucination_batch_summary.json` is written by default.

## Inputs

Each `sample_*` directory must contain:

- `result.json`, with `structured_report.findings` and `structured_report.impressions`
- `summary.json`, with `ground_truth`
- optional `images/`, included as evidence manifest

Image evidence is enabled by default for multimodal models. The script sends the first raster images, together with descriptions from each case `image_index.json`, to Stage A support prompts and the final taxonomy prompt.

Useful controls:

```bash
--max-llm-images 4 --image-stage all
```

For text-only models, disable image upload:

```bash
--no-include-images
```

## Batch Controls

```bash
python -m hallucination_detection \
  --target-dir output/mimic/full_radclaw \
  --model YOUR_MODEL_NAME \
  --limit 5 \
  --fail-fast
```

Use `--no-json-mode` for OpenAI-compatible endpoints that do not support `response_format={"type":"json_object"}`.

The runner prints per-case and per-stage progress by default, including Stage 0 extraction, Stage A support verification, Stage B CRIMSON matching, Stage C checks, and final taxonomy classification. Use `--quiet` to suppress progress logs.

Cases are skipped by default when `sample_xxx/hallucination_detection.json` already exists. Use `--overwrite` to force re-evaluation.
