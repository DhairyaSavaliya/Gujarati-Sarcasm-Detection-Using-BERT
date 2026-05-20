from transformers import Trainer, TrainingArguments
from .model import load_model_and_tokenizer
from .dataset import load_dataset
from .config import MODEL_SAVE_DIR

def train_model():
    tokenizer, model = load_model_and_tokenizer()
    dataset = load_dataset()

    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            padding="max_length",
            truncation=True,
            max_length=64
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    split = dataset.train_test_split(test_size=0.2)
    train_ds = split["train"]
    test_ds = split["test"]

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_steps=50
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds)
    trainer.train()

    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
