# End-to-End Validation Report

## Part D: Kubernetes Training Job

### Manifests Applied

```bash
kubectl apply -f k8s/namespace.yaml
# namespace/ml-training created

kubectl apply -f k8s/configmap.yaml
# configmap/training-config created

kubectl apply -f k8s/pvc.yaml
# persistentvolumeclaim/data-pvc created
# persistentvolumeclaim/checkpoints-pvc created

kubectl apply -f k8s/training-job.yaml
# job.batch/model-training created
```

### Job Completion

```text
NAME                   READY   STATUS      RESTARTS   AGE
model-training-vx8jc   0/1     Completed   0          161m
```

The Kubernetes training Job completed successfully with no container restarts.

### Training Logs

```json
{"event": "device_selected", "device": "cpu"}
{"epoch": 1, "train_loss": 1.362, "train_accuracy": 0.5034, "val_loss": 1.172, "val_accuracy": 0.6133}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 2, "train_loss": 0.88, "train_accuracy": 0.6888, "val_loss": 0.9553, "val_accuracy": 0.6783}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"event": "training_complete", "best_val_loss": 0.9553}
```

The initial Kubernetes validation run trained for two epochs on CPU and successfully wrote the model checkpoint to the shared checkpoints volume.

---

## Part E: Kubernetes Serving

### Deployment Status

```text
NAME                            READY   STATUS    RESTARTS   AGE
model-serving-89695d9cc-gnr2l   1/1     Running   0          48s
model-serving-89695d9cc-hv9rw   1/1     Running   0          48s
```

Both serving replicas reached the `Running` and `Ready` states successfully.

### Deployment Configuration

- Replicas: **2/2 available**
- Deployment strategy: **RollingUpdate**
  - `maxSurge: 1`
  - `maxUnavailable: 0`
- Liveness probe:
  - Endpoint: `GET /health`
  - Period: every 10 seconds
  - `failureThreshold: 3`
- Readiness probe:
  - Endpoint: `GET /health`
  - Period: every 5 seconds
  - `initialDelaySeconds: 15`
- Model checkpoint storage:
  - Shared checkpoints PVC mounted by the serving pods
  - Mounted read-only in the serving containers

---

## Part F: End-to-End Validation

### Health Check

Command:

```bash
curl http://localhost:8080/health
```

Response:

```json
{"status":"ok"}
```

The health endpoint confirmed that the inference service was available and healthy.

### Prediction Test

Command:

```bash
curl -X POST http://localhost:8080/predict   -F "image=@test_image.png"
```

Response:

```json
{
  "predicted_class": "airplane",
  "predicted_index": 0,
  "probabilities": {
    "airplane": 0.5452,
    "automobile": 0.0,
    "bird": 0.09,
    "cat": 0.0029,
    "deer": 0.0499,
    "dog": 0.0001,
    "frog": 0.0005,
    "horse": 0.0001,
    "ship": 0.3108,
    "truck": 0.0005
  }
}
```

The prediction endpoint successfully accepted an image, performed inference using the trained checkpoint, and returned the predicted class together with class probabilities.

### Supplementary Training: Checkpoint Resume

After checkpoint-resume support was added in **PR #7**, a supplementary training run successfully resumed from epoch 5 and reached **82.03% validation accuracy**.

This supplementary run demonstrated:

- restoration of the saved model state;
- restoration of optimizer state;
- continuation from the correct training epoch;
- reuse of an existing checkpoint instead of restarting from random initialization; and
- improved model quality beyond the initial two-epoch Kubernetes validation run.

---

## Reflection

The most challenging part of this assignment was not the PyTorch model or the Kubernetes manifests themselves, because those followed fairly standard patterns once designed. The real challenge was maintaining environment reliability and managing resources across multiple machines with different constraints, while diagnosing genuine mistakes under time pressure.

I initially worked on a laptop with 7.6 GB of RAM. This proved insufficient once Minikube, a training container, and other background applications were running simultaneously, resulting in repeated out-of-memory terminations. Diagnosing the issue required checking actual memory and process usage instead of assuming that Kubernetes or Docker was malfunctioning.

A second machine presented a different limitation: disk space. The problem was aggravated by the fact that the default PyTorch package from PyPI can pull large NVIDIA CUDA-related libraries even when training is performed on a CPU-only system. Because this machine also hosted production infrastructure, cleanup had to be performed carefully. Rather than risk affecting existing services, I eventually isolated the assignment workload using a second Docker daemon backed by separate storage.

I also encountered two significant Git and source-code mistakes. A pull request intended for `develop` was accidentally merged into `main`, causing the branches to diverge. Instead of rewriting repository history, I traced the problem to a stale local branch and corrected it using a normal forward merge. Separately, `train.py` was accidentally overwritten with the contents of a test file. The resulting container image still built successfully, but the container exited immediately without performing training. This reinforced an important lesson: a successful build or clean diff does not guarantee correct runtime behavior.

Another gap became apparent during longer training runs. The original training script did not support checkpoint resume, meaning that a restarted training run discarded previous progress and initialized a new model. I therefore added logic to restore the saved model and optimizer states and continue from the correct epoch. This significantly improved the practicality of repeated training, both locally and within the Kubernetes workflow.

I also attempted the optional GPU bonus. The NVIDIA driver was installed, but Secure Boot prevented the kernel module from loading unless it was trusted through Machine Owner Key enrollment. Completing that enrollment requires physical keyboard confirmation during boot and could not be performed remotely, so GPU-accelerated training was deferred.

Once the environment was stable, the Kubernetes workflow behaved as intended. The training Job wrote a checkpoint to a shared PersistentVolumeClaim, and the serving Deployment consumed that checkpoint through the same persistent storage. The readiness probes correctly reflected whether a usable checkpoint was available. Observing the serving pods become healthy once the checkpoint appeared demonstrated the practical value of Kubernetes-native persistent storage for ML pipelines.

Overall, the main lesson from the assignment was that MLOps work is often dominated by infrastructure and environment management—memory, disk space, SELinux permissions, Secure Boot, process isolation, and persistent storage—rather than by the machine learning model itself.

---

## Notes

- **GPU bonus (Part D, +5 pts):** Attempted but blocked by Secure Boot on the available GPU machine. The NVIDIA driver was installed and a MOK key was staged, but enrollment requires physical keyboard access during boot. GPU validation was therefore deferred.
- **Model checkpoint:** The epoch-5 checkpoint achieving **82.03% validation accuracy** is tracked using DVC.
- **DVC remote:** Google Drive is used as the DVC remote to support reproducibility and checkpoint transfer across machines.
