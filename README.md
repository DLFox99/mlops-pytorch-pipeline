# Assignment 03 on Machine Learning Operations
## Topic: Deploying PyTorch ML Workloads with Docker & Kubernetes

# mlops-pytorch-pipeline

CIFAR-10 image classifier deployed through a full MLOps lifecycle: local
training, Docker containerization, and Kubernetes orchestration for both
training and serving.

Built for DA5402W / ID5003W — Assignment 03 (Deploying PyTorch ML Workloads
with Docker & Kubernetes).

## Overview
Trains a ResNet-18 on CIFAR-10 (adapted for the small 32x32 images since
stock ResNet-18 expects 224x224). Config lives in a YAML file so you're not
hardcoding hyperparameters, and training bails early if val loss stops
improving instead of grinding through all epochs for no reason. Logs come
out as JSON lines because parsing plain text logs later is a pain.

Serving is a small FastAPI app. Two routes: /predict takes an image and
gives back probabilities, /health tells you if the model actually loaded
(useful for k8s probes, not just a formality).

Docker images are split: one for training, one for serving. Training image
has all the heavy stuff (torch, torchvision, etc). Serving image is kept
lean, no reason to ship tensorboard and friends to a prod container that
just runs inference.

On the k8s side: training runs as a Job since it's a run-once-and-die kind
of workload, serving runs as a Deployment since it needs to stay up, plus
a Service in front of it so it's actually reachable.

## Prerequisites

- Python 3.11+
- Docker
- kubectl
- A local Kubernetes cluster (Minikube or kind)

## Project structure

```
mlops-pytorch-pipeline/
├── src/            # model, dataset, train, and code
├── configs/        # hyperparameters for training
├── docker/         # Dockerfile.train, Dockerfile.serve
├── k8s/            # Kubernetes Related
├── requirements/   # dependencies
├── tests/          # unit tests
└── .github/workflows/  # CI
```


## Running locally

```bash
pip install -r requirements/train.txt
python src/train.py
```

Pulls hyperparams from `configs/training_config.yaml` by default.

## Running with Docker

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints mlops-train:v1

docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 -v $(pwd)/checkpoints:/app/checkpoints mlops-serve:v1

curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

(On SELinux systems like Fedora, add `:z` to the volume mounts above or
you'll hit permission errors.)

## Running on Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
```

## Git workflow

- `main`: stable
- `develop`: integration branch
- `feature/*`: actual work happens here, merged via PR
- **Conventional Commits** (feat, fix, chore, docs, ci, etc.)
