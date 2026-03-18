## MediTriage‑LLM: Fine‑tuned Triage Model

This repo fine‑tunes `mistralai/Mistral-7B-Instruct-v0.3` with QLoRA on a medical triage dataset so that, given a free‑text patient portal message, the model outputs a structured JSON object:

```json
{
  "department": "Orthopedics",
  "symptoms": ["swollen left ankle", "warm to touch", "throbbing pain", "inability to bear weight"],
  "condition": "Unknown",
  "sentiment": "Anxious",
  "urgency_level": "High"
}
```

### Python files overview

- **`generate_triage_dataset.py`**
  - **Purpose**: Generate a large synthetic triage dataset (patient messages + labels) using external LLMs via the OpenRouter API.
  - **Input**: None (talks to API, writes to `data/raw/`).
  - **Output**:
    - `data/raw/symptom_clusters.json`
    - `data/raw/raw_dataset.json`
    - `data/raw/final/*.json` (if Stage 3 is run).

- **`prepare_dataset.py`**
  - **Purpose**: Convert a CSV of patient questions + target JSONs into instruction‑tuning format and split into train/val/test.
  - **Input**:
    - `data/raw/finetuning_data.csv` with columns:
      - `questions`: free‑text patient message
      - `output`: JSON string with keys like `department`, `symptoms`, `condition`, `sentiment`, `urgency_level`
  - **Output**:
    - `data/processed/train.json`
    - `data/processed/val.json`
    - `data/processed/test.json`
    - Each record has:
      - `messages`: chat template with system + patient message and JSON answer
      - `ground_truth`: parsed target JSON
      - `patient_message`

- **`finetune.py`**
  - **Purpose**: Fine‑tune Mistral‑7B using QLoRA/PEFT on the processed dataset.
  - **Input**:
    - `data/processed/train.json`
    - `data/processed/val.json`
  - **Output**:
    - LoRA adapter + tokenizer in `models/meditriage-mistral-lora/`
    - Training logs in `models/logs/training_log.json`
    - Loss plot in `data/outputs/plots/train_val_loss.png`
  - **Key details**:
    - Base model: `mistralai/Mistral-7B-Instruct-v0.3` (4‑bit quantized).
    - QLoRA with `r=16`, `lora_alpha=32`, `lora_dropout=0.05` on attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
    - Only ~0.19% of parameters are trainable (LoRA adapters); the base model is frozen.

- **`inference_base.py`**
  - **Purpose**: Run the **base** (un‑fine‑tuned) Mistral model on the test set to get baseline predictions.
  - **Input**:
    - `data/processed/test.json`
  - **Output**:
    - `data/outputs/base_model_predictions.json`
    - Each entry contains:
      - `patient_message`
      - `ground_truth`
      - `raw_model_output` (string from the model)
      - `parsed_output` (parsed JSON or `null` if parsing failed)
      - `valid_json` (boolean).

- **`inference_finetuned.py`**
  - **Purpose**: Run the **fine‑tuned** (base + LoRA adapter) model on the test set.
  - **Input**:
    - `data/processed/test.json`
    - `models/meditriage-mistral-lora/` (LoRA adapter + tokenizer)
  - **Output**:
    - `data/outputs/finetuned_model_predictions.json`
    - Same structure as the base predictions file, for direct comparison.


### How to fine‑tune (Colab or local GPU)

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Prepare the dataset**

Place your `finetuning_data.csv` under `data/raw/`, then run:

```bash
python prepare_dataset.py
```

3. **Run fine‑tuning**

```bash
python finetune.py
```

This:
- Trains QLoRA adapters on `train.json`
- Uses `val.json` to compute validation loss after training
- Saves:
  - Fine‑tuned adapter in `models/meditriage-mistral-lora/`
  - Training metrics in `models/logs/training_log.json`
  - Train vs val loss plot in `data/outputs/plots/train_val_loss.png`

4. **Run baseline and fine‑tuned inference on test**

```bash
python inference_base.py       # base model on test.json
python inference_finetuned.py  # fine‑tuned model on test.json
```

Outputs:
- `data/outputs/base_model_predictions.json`
- `data/outputs/finetuned_model_predictions.json`

### How to load `models/meditriage-mistral-lora` and run custom evaluation

- Clone the repo
- Place the fine‑tuned adapter folder at `models/meditriage-mistral-lora/`
- Plase the evaluation dataset in `data/processed/`
```



