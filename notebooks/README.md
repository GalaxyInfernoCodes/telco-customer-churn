# Demo Notebooks — Setup Guide

Four notebooks for the three demos in the MLOps lecture.

| File | Demo | Time |
|---|---|---|
| `demo_1_mlflow_tracking.py` | Demo 1 — MLflow tracking | ~3.5 min live |
| `demo_2_delta_tables.py` | Demo 2 — Reading/writing Delta tables | ~3.5 min live |
| `demo_3a_feature_prep.py` | Demo 3 — Workflow Task A (Silver layer) | ~3 min live |
| `demo_3b_batch_inference.py` | Demo 3 — Workflow Task B (predictions) | (same demo) |

All four are in **Databricks Python source format**. They import directly as notebooks — no conversion needed.

---

## How to import into Databricks Free Edition

1. Sign in to [Databricks Free Edition](https://www.databricks.com/learn/free-edition)
2. Workspace sidebar → **Workspace** → your home folder
3. Right-click → **Import** → choose **File**
4. Drop in one or more `.py` files. Databricks recognizes the `# Databricks notebook source` header and creates real notebooks (not Python files)
5. Repeat for all four

Recommended folder structure inside your workspace:
```
mlops_lecture/
├── demo_1_mlflow_tracking
├── demo_2_delta_tables
└── demo_3_workflow/
    ├── feature_prep   (= demo_3a)
    └── batch_inference (= demo_3b)
```

---

## Setting up the Demo 3 workflow

After importing both `demo_3a` and `demo_3b`:

1. Sidebar → **Workflows** → **Create job**
2. Job name: `demo_batch_inference_pipeline`
3. **Task 1**:
   - Name: `feature_prep`
   - Type: Notebook
   - Source: Workspace
   - Path: select your imported `demo_3a_feature_prep` notebook
   - Compute: serverless (default in Free Edition)
4. **Task 2**:
   - Name: `batch_inference`
   - Type: Notebook
   - Path: select your imported `demo_3b_batch_inference` notebook
   - **Depends on**: `feature_prep` ← this is the key field
5. Optional polish that students will ask about:
   - **Schedule** → Add trigger → Scheduled → "Daily at 02:00 UTC"
   - **Notifications** → On failure → your email
   - **Retries** → On failure: 2 retries, 5 min delay (set per-task)
6. **Save** → click **Run now** to verify both tasks succeed

The DAG view will show two boxes connected by an arrow. That's your demo visual.

---

## Pre-class smoke test (do this the day before)

Run each notebook end-to-end once. In Free Edition, the cluster cold-start can take 1–2 minutes — running them once warms things up and lets you spot anything that's changed (e.g., if `samples.nyctaxi.trips` schema ever shifts).

For Demo 1, after running, open the Experiments panel and confirm both `rf_small` and `rf_big` runs show up. Then click "Compare" to make sure the comparison view loads — that's the moment you'll show students.

For Demo 3, click **Run now** on the workflow once before class so its run history isn't empty. A history with 3–5 successful runs (and ideally one failed run from when you broke something on purpose) tells the story way better than a fresh job.

---

## Backup recordings

Once you've smoke-tested, do one screen recording per demo:
- macOS: `Cmd+Shift+5` → Record selected portion
- Loom or OBS work too

Save as `demo_1.mp4`, `demo_2.mp4`, `demo_3.mp4` and have them open in tabs. If Free Edition's cluster takes too long during the live class, switch to the recording without ceremony — students won't notice.

---

## Datasets used

All demos use built-in data — no uploads, no external dependencies:

- **Demo 1**: `sklearn.datasets.load_breast_cancer` (ships with sklearn, ~600 rows)
- **Demo 2 & 3**: `samples.nyctaxi.trips` (built into every Databricks workspace, ~21k rows)

The Demo 2/3 tables write to `workspace.default.demo_*` — the default writable schema in Free Edition. If you've renamed your default schema, update the table paths in cells 3 (Demo 2), the write block in 3a, and the read in 3b.
