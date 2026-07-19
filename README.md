# Consumer Complaint Classification (NLP)

Classifies customer complaint narratives into 5 categories: `credit_card`,
`credit_reporting`, `debt_collection`, `mortgages_and_loans`, `retail_banking`.

## Files
- `notebooks/Consumer_Complaint_Classification.ipynb` — full pipeline: EDA, preprocessing,
  imbalance handling, tokenization/padding, 4 models (SimpleRNN, LSTM, GRU, Transformer),
  evaluation, comparison, and a Gradio demo cell.
- `prep.py` — text cleaning (lowercase, strip punctuation/numbers, stopwords, lemmatization).
- `model_utils.py` — model builders (RNN/LSTM/GRU + a from-scratch Transformer encoder) and
  the evaluation function (accuracy/precision/recall/F1/confusion matrix).
- `train.py` — loads data, samples/cleans it, builds tokenizer + padded sequences, saves splits.
- `train_one.py NAME` — trains and evaluates one model (`SimpleRNN`, `LSTM`, `GRU`, or
  `TransformerEncoder`) and saves it to `models/`.
- `aggregate_results.py` — combines all 4 models' metrics, picks the best by F1, saves
  `models/results_comparison.csv` and `models/model_comparison.png`.
- `app.py` — Gradio web app; loads the best saved model and classifies new complaints with a
  confidence score.
- `models/` — trained models (`.keras`), tokenizer, label encoder, metrics, and plots.

## How to run
```bash
pip install -r requirements.txt

python train.py                       # prepares data (adjust SAMPLE_PER_CLASS / MAX_LEN as needed)
python train_one.py SimpleRNN
python train_one.py LSTM
python train_one.py GRU
python train_one.py TransformerEncoder
python aggregate_results.py           # picks the best model

python app.py                         # launches the Gradio app
```

Or just open and run the notebook top to bottom.

## Important note: HuggingFace Transformer
Section 6.4(a) of the notebook contains the exact code to fine-tune a real pretrained
HuggingFace model (`distilbert-base-uncased`). It was **not executed here** because this
environment has no internet access to `huggingface.co`, so pretrained weights can't be
downloaded. Run that cell on Google Colab / Kaggle / your own machine (ideally with a GPU) to
get the actual fine-tuned-HF result. In its place, a Transformer **encoder built from scratch**
in Keras (multi-head self-attention + positional embeddings) was trained end-to-end here so the
notebook has real results and a working deployed app — it turned out to be the
best-performing model on this data.

## Scaling up
Everything was trained on a capped, stratified sample (2,500 complaints/class, ~12.5k rows,
sequence length 80, 4 epochs) purely because this sandbox is CPU-only. The full cleaned
dataset has ~124k rows. To improve accuracy for your final submission: increase
`SAMPLE_PER_CLASS` in `train.py` (or remove the cap), increase `MAX_LEN`/`EPOCHS`, and train on
a GPU (Colab is free) — no other code changes are required.

## Publishing this project to GitHub

1. Create a new **empty** repo on GitHub (no README/.gitignore) at
   `https://github.com/new`, e.g. named `consumer-complaint-classification`.

2. From inside this project folder:
   ```bash
   git init
   git add .
   git status   # data/*.csv and models/*.keras should NOT show up (they're in .gitignore)
   git commit -m "Initial commit: NLP consumer complaint classification project"
   git branch -M main
   git remote add origin https://github.com/USERNAME/consumer-complaint-classification.git
   git push -u origin main
   ```
   Replace `USERNAME` with your GitHub username. If asked for a password, use a
   **Personal Access Token** instead (Settings → Developer settings → Personal access tokens →
   Generate new token, scope `repo`).

3. The raw dataset (`data/*.csv`) and trained model weights (`models/*.keras`) are excluded
   via `.gitignore` because they're large binary files GitHub doesn't handle well in normal
   commits. To include them anyway:
   - **Easiest:** upload them as attachments on a GitHub **Release** instead of committing them.
   - **Or use Git LFS** to track them like normal files:
     ```bash
     git lfs install
     git lfs track "*.keras"
     git add .gitattributes
     git add models/*.keras
     git commit -m "Add trained models via Git LFS"
     git push
     ```
