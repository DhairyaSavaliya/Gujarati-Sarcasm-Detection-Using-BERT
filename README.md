# Gujarati Sarcasm Detection Using BERT (MuRIL)

An NLP project aimed at detecting sarcasm in Gujarati text using **MuRIL (Multilingual Representations for Indian Languages)**, a BERT model pretrained specifically on Indian languages. This repository includes preprocessing scripts utilizing the **Indic NLP Library**, model training, evaluation, and a prediction interface.

---

## 🚀 Features

- **Indic NLP Normalization**: Standardizes Gujarati text, removing dialectal variations and formatting anomalies.
- **Trivial Tokenization**: Custom tokenization tailored for Gujarati grammar and structure.
- **MuRIL Fine-Tuning**: Leverages the power of `google/muril-base-cased` for sequence classification, fine-tuned specifically for binary sarcasm detection.
- **Unified CLI Entrypoint**: Easily run preprocessing, model training, and prediction from a single terminal interface.

---

## 📁 Repository Structure

```text
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration file defining model parameters and data paths
│   ├── dataset.py          # PyTorch / Hugging Face datasets utility
│   ├── model.py            # Loads pre-trained MuRIL model and tokenizer
│   ├── preprocess.py       # Normalization and tokenization pipelines for Gujarati
│   ├── train.py            # Trainer API setup, training, and evaluation logic
│   └── predict.py          # Inference script to evaluate sarcasm on new sentences
├── main.py                 # Command Line Interface (CLI) entrypoint
├── requirements.txt        # Core project dependencies
└── .gitignore              # Files/folders excluded from version control
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/DhairyaSavaliya/Gujarati-Sarcasm-Detection-Using-BERT.git
cd Gujarati-Sarcasm-Detection-Using-BERT
```

### 2. Set Up Virtual Environment (Optional but Recommended)
```bash
python -m venv sarcasm_env
# On Windows:
sarcasm_env\Scripts\activate
# On Linux/macOS:
source sarcasm_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Indic NLP Library Resources
This project utilizes the `indicnlp` library locally. Create the resources directory and clone the necessary packages:
```bash
mkdir indic_resources
cd indic_resources

# Clone the Indic NLP Library
git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git

# Clone the Indic NLP Resources
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git

cd ..
```

---

## 📊 Dataset Setup

Before preprocessing and training, place your Gujarati sarcasm dataset in the following directory layout:
```text
data/
└── raw/
    └── sarcasm_data.csv
```

The CSV dataset should have the headers `text` and `label` (or no headers, with columns corresponding to `text,label`), where:
- `text`: The Gujarati sentence.
- `label`: `1` (or "sarcasm", "true", "yes") for sarcasm, and `0` (or "not_sarcasm", "false", "no") for non-sarcastic text.

---

## 🖥️ Usage

You can execute all steps using `main.py`:

### Step 1: Preprocess Data
Normalizes, tokenizes, and saves the cleaned dataset to `data/processed/cleaned_data.csv`.
```bash
python main.py preprocess
```

### Step 2: Train the Model
Splits the preprocessed dataset into train/test splits, tokenizes for MuRIL, trains the model for 3 epochs, and saves the fine-tuned checkpoint to `models/muril_sarcasm/`.
```bash
python main.py train
```

### Step 3: Sarcasm Prediction
Run inference on a single Gujarati sentence.
```bash
python main.py predict "તમારું લખાણ અહીં લખો" # Enter your text here
```

Example prediction command:
```bash
python main.py predict "અરે વાહ! તમે તો બહુ જ હોશિયાર છો."
```

---

## 🧠 Model Details: MuRIL

**MuRIL (Multilingual Representations for Indian Languages)** is a BERT-based transformer model developed by Google, pre-trained on 17 monolingual Indian languages (including Gujarati) and their transliterated counterparts. 

- **Pre-trained model**: `google/muril-base-cased`
- **Tokenizer**: `BertTokenizer`
- **Output Layer**: Sequence classification head with 2 classes (Sarcastic / Not Sarcastic)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
