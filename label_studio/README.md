# Pillar 3 — Human labels & QA (Label Studio)

Use this when you need **gold labels**, adjudication, or inter-annotator agreement on claims (X-Fact–style or your own binarized scheme).

## Quick start

```bash
pip install label-studio
label-studio start --port 8080
```

1. Create a project → **Settings → Labeling Interface → Code** → paste `config/claim_verification.xml`.
2. **Import** `sample_tasks/sample_tasks.json` (replace with your JSONL/CSV export from the notebook pipeline).

## Export

- **Export** annotations as JSON or CSV for training / error analysis in the fine-tune notebooks.

## Resume

- *Built **Label Studio** annotation specs and review rounds for multilingual claim verification.*
