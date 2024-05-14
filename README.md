# 🏃‍➡️ Segment3d Runner

Worker service for 3D gaussian splatting and segmentation pipeline.

## Cloning the repository

This repository contains submodules for the models used in the pipeline. Run this command to also clone the submodules:

```bash
git clone --recurse-submodules https://github.com/segment3d-app/segment3d-runner.git
```

## Start the Kubernetes pod

Make sure to run the following commands below inside a Kubernetes pod, to ensure a clean working environment. To start the pod, run this command:

```bash
kubectl apply -f pod.yaml
```
