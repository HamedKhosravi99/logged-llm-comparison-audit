# Security and privacy

## What this Space does with uploads

Uploaded tables are read into memory, analysed, and written to a per-run temporary directory
that the operating system reclaims. The application:

- makes **no network calls** of any kind (enforced by a test that walks the import graph);
- requires **no API token or secret**;
- never writes uploaded rows to stdout, stderr, or any log;
- never copies uploads into the repository;
- uses **no persistent storage** and no Storage Bucket, so nothing survives a restart;
- disables Gradio analytics (`analytics_enabled=False`).

The default Spaces disk is ephemeral. Do not attach persistent storage to this demonstration.

## What you should not upload

Do not upload confidential conversations, personal information, credentials, or proprietary
prompts. This is a public demonstration. Only the columns in the documented schemas are read;
supply aggregated or synthetic traces.

## Running on confidential data

Run the package locally instead:

```bash
pip install -e .
python -c "import pandas as pd; from routeraudit import AuditConfig, analyze_trace; ..."
```

or build the Docker image and keep the data inside your own environment.

## Reporting an issue

Open an issue in the source repository. Do not include customer data in the report.
