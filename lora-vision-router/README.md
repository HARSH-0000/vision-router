# Multi-Adapter LoRA Vision Router

A production-ready, multi-tenant image classification API using a single frozen Vision Transformer (`google/vit-base-patch16-224`) paired with hot-swappable LoRA adapters (~1.5 MB each) to eliminate catastrophic forgetting across domains.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train isolated domain adapters on CIFAR-10:
   ```bash
   python -m src.train_adapter --domain vehicles --epochs 2
   python -m src.train_adapter --domain animals --epochs 2
   ```

3. Run the zero-forgetting regression suite:
   ```bash
   pytest -v
   ```

4. Start the FastAPI inference server:
   ```bash
   uvicorn app.main:app --reload
   ```
