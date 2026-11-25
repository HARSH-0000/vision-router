# Multi-Adapter LoRA Vision Router

A CIFAR-10 image-classification demo that uses one shared Vision Transformer (ViT) with separate, switchable LoRA adapters for vehicle and animal classes.

> This is a research/learning project, not a production-ready multi-tenant service. Accuracy depends on training settings and hardware; no accuracy threshold is guaranteed.

## Architecture

- Base model: `google/vit-base-patch16-224`
- Dataset: CIFAR-10
- Domains:
  - `vehicles`: airplane, automobile, ship, truck
  - `animals`: bird, cat, deer, dog, frog, horse
- Adapter method: PEFT LoRA
- API: FastAPI

Both adapters use the same 10-output classifier and retain CIFAR-10's original class IDs. Inference restricts the output scores to the requested domain. This shared classifier shape is important: adapters can then be loaded into the same base model without classifier dimension mismatches.

For the ViT implementation used here, LoRA targets the attention projection modules named `query` and `value`.

## Requirements

- Python 3.11
- CPU or CUDA-capable PyTorch environment
- Internet access on first run to download the pretrained ViT and CIFAR-10

Install the pinned dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a CPU-only PyTorch installation, install the matching CPU wheels first:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Train adapters

Run commands from the `lora-vision-router` directory:

```bash
python -m src.train_adapter --domain vehicles --epochs 2
python -m src.train_adapter --domain animals --epochs 2
```

Optional arguments include `--batch-size`, `--learning-rate`, `--seed`, `--data-dir`, and `--output-dir`.

Each command saves its adapter under `adapters/<domain>/`. The API expects both adapter directories to contain an `adapter_config.json`; train both domains before making prediction requests.

## Run tests

```bash
python -m pytest -q
```

Unit tests run without trained adapters. The adapter-isolation integration test is skipped until both adapters have been trained. To run only the integration test after training:

```bash
python -m pytest -m integration -v
```

The integration test checks that loading the second adapter does not change the first adapter's accuracy. It does not claim that either adapter reaches a particular accuracy.

## Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for interactive API documentation.

### Predict

Send a multipart image upload with a `domain` query parameter:

```bash
curl -X POST "http://localhost:8000/predict?domain=vehicles" \
  -F "file=@sample.jpg"
```

Use `domain=animals` for the animal adapter. The API returns the domain, predicted class, CIFAR-10 class ID, confidence, and latency.

The endpoint rejects unknown domains, non-image content types, invalid images, empty uploads, and files larger than 10 MB. Model and adapter loading is lazy, so importing the FastAPI app does not download model weights. Requests are serialized around adapter selection and inference because adapter selection mutates shared model state.

## Docker

Build from the repository root:

```bash
docker build -f lora-vision-router/Dockerfile -t lora-vision-router ./lora-vision-router
```

The image contains the application but not trained adapter weights. To run it, mount the locally trained adapters:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/lora-vision-router/adapters:/app/adapters" \
  lora-vision-router
```

On PowerShell, replace the volume path with the equivalent absolute Windows path.

## Project structure

```text
lora-vision-router/
├── .github/workflows/ci.yml
├── adapters/
├── app/main.py
├── data/                 # downloaded CIFAR-10 data (git-ignored)
├── src/dataset.py
├── src/train_adapter.py
├── tests/
│   ├── test_dataset.py
│   └── test_forgetting.py
├── Dockerfile
└── requirements.txt
```
