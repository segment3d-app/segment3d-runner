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
kubectl exec -it segment3d-pod -- /bin/bash
```

## Configure the models

Before running the model setups, initialize environment:

```bash
conda init
export DEBIAN_FRONTEND=noninteractive
```

Each models used in the pipeline have their own requirements and configurations. Setup scripts are provided to initialize the models, which can be accessed in the `/setups` folder. To setup all models, follow this step:

```bash
cd setups
chmod +x setup.sh
bash setup.sh
```

## Install dependencies

Before running the main script, install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Running main script

Update the access control of the main script to be runnable:

```bash
chmod u+x ./src/main.py
```

To run the main script, you can run it directly:

```bash
python ./src/main.py
```

Or run in the background with nohup:

```bash
nohup python ./src/main.py &
```
