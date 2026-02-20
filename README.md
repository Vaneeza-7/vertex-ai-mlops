# Vertex AI MLOps Learning Path

This repository contains a structured, hands-on implementation of Vertex AI Pipelines from fundamentals to a production-style custom container workflow (Phase 0–3).

It is designed to move from orchestration basics to managed training, model registry, and controlled deployment.

---

## Scope

Covers:

* Pipeline orchestration (DAG lifecycle)
* Lightweight Python components
* Artifact-based data flow (Dataset, Model, Metrics)
* Managed training with Vertex Custom Jobs
* Docker-based training and serving containers
* Model Registry integration
* Conditional deployment to Vertex Endpoints

---

## Phase Summary

### Phase 0 — Infrastructure Setup

* GCP project and single-region configuration
* GCS bucket as pipeline root
* Dedicated service account with minimal IAM
* Vertex AI Workbench setup

Result: A minimal pipeline runs successfully.

---

### Phase 1 — Pipeline Fundamentals

* Define → compile → run lifecycle
* Component vs task distinction
* DAG visualization in Vertex UI
* Logs and artifact inspection

Result: Clear understanding of pipeline orchestration.

---

### Phase 2 — Artifact-Based ML Pipeline

* Lightweight components using Dataset, Model, and Metrics artifacts
* Artifact path handling inside components
* Lineage and reproducible runs in Vertex UI

Result: Functional ML pipeline with tracked artifacts.

---

### Phase 3 — Custom Container Production Workflow

* Build training and serving Docker images
* Push images to Artifact Registry
* Launch managed training via Vertex Custom Job
* Upload model to Model Registry
* Apply metric threshold gate
* Deploy to Vertex Endpoint conditionally

Result: End-to-end train → evaluate → register → deploy pipeline with production safeguards.

---

## Repository Structure

```text
pipelines/
  pipeline.py

components/
  preprocess.py
  evaluate.py

training/
  train.py

serving/
  app.py

infra/
  compile.sh
  run.py

Dockerfile.train
Dockerfile.serve
```

---

## Prerequisites

* GCP project with Vertex AI enabled
* Regional GCS bucket (pipeline root)
* Artifact Registry Docker repository
* Dedicated pipeline service account
* Vertex AI Workbench (recommended)

---

## Execution Flow

1. Compile pipeline to YAML
2. Submit run using PipelineJob
3. Monitor DAG and logs in Vertex AI
4. Inspect:

   * Artifacts in GCS
   * Model in Model Registry
   * Deployment in Endpoints

---

## Outcome

By completing this repository, you will be able to:

* Build reproducible ML pipelines
* Use artifact-driven step communication
* Run managed training jobs
* Deploy custom container models
* Gate production deployments using metrics

This project represents a clean progression from pipeline basics to production-oriented MLOps on Vertex AI.
