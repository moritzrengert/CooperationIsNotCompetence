# Data input

The anonymized observations are released separately in the companion
`publication_dataset` Hugging Face package. Copy or download its ORS files into
this directory, or point analysis scripts at a local output directory with
`LLM_PGG_OUTPUTS`.

Expected analysis inputs include:

- `analysis/clean_decisions.csv`
- `analysis/clean_group_periods.csv`
- `analysis/model_metadata.csv`

The repository contains no logs, raw model responses, reasoning text, API
identifiers, credentials, or local filesystem paths.
