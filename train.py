# train.py

import argparse
import torch
from datasets import Dataset
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

def load_data(lug_path, eng_path):
    """Loads parallel text files into a Hugging Face Dataset."""
    with open(lug_path, 'r', encoding='utf-8') as f:
        lug_lines = [line.strip() for line in f.readlines()]
    with open(eng_path, 'r', encoding='utf-8') as f:
        eng_lines = [line.strip() for line in f.readlines()]

    if len(lug_lines) != len(eng_lines):
        raise ValueError("The number of lines in Luganda and English files do not match.")

    # Create a dictionary in the format Hugging Face's Dataset expects
    data = {"translation": []}
    for i in range(len(lug_lines)):
        data["translation"].append({"en": eng_lines[i], "lg": lug_lines[i]})

    # Convert to a Hugging Face Dataset object
    return Dataset.from_dict(data)

def load_data_from_excel(luganda_excel_path, english_excel_path=None):
    """Load translation data directly from Excel files."""
    try:
        # Import pandas here to avoid requiring it for text-based loading
        import pandas as pd
        
        print(f"Loading data from Excel files...")
        # Check if it's a single file or two files
        if english_excel_path is None:
            # Single Excel file with both languages
            df = pd.read_excel(luganda_excel_path)
            # Assuming columns are named "Luganda" and "English"
            # Modify these column names if your Excel structure is different
            if "Luganda" in df.columns and "English" in df.columns:
                lug_lines = df["Luganda"].astype(str).tolist()
                eng_lines = df["English"].astype(str).tolist()
            else:
                # Use column C (index 2) as specified
                print("Using column C (index 2) for data extraction")
                if len(df.columns) > 2:
                    lug_lines = df.iloc[:, 2].astype(str).tolist()
                    # Assuming English is in the next column
                    eng_lines = df.iloc[:, 3].astype(str).tolist() if len(df.columns) > 3 else []
                else:
                    raise ValueError("Excel file does not have enough columns (need column C)")
        else:
            # Two separate Excel files
            df_lug = pd.read_excel(luganda_excel_path)
            df_eng = pd.read_excel(english_excel_path)
            
            # Use column C (index 2) for both files as specified
            if len(df_lug.columns) <= 2:
                raise ValueError("Luganda Excel file does not have column C (index 2)")
            if len(df_eng.columns) <= 2:
                raise ValueError("English Excel file does not have column C (index 2)")
                
            lug_lines = df_lug.iloc[:, 2].astype(str).tolist()
            eng_lines = df_eng.iloc[:, 2].astype(str).tolist()
        
        print(f"Loaded {len(lug_lines)} Luganda lines and {len(eng_lines)} English lines")
        
        # Filter out nan values and empty strings
        data = {"translation": []}
        for i in range(min(len(lug_lines), len(eng_lines))):
            lg, en = lug_lines[i].strip(), eng_lines[i].strip()
            if lg != "nan" and en != "nan" and lg and en:
                data["translation"].append({"en": en, "lg": lg})
        
        print(f"Created dataset with {len(data['translation'])} valid translation pairs")
        return Dataset.from_dict(data)
    except Exception as e:
        print(f"Error loading Excel files: {e}")
        raise

def main(args):
    # --- 1. Load Model and Tokenizer ---
    # The model name for NLLB-200, 600M parameter version.
    # It's a good balance of size and performance.
    MODEL_NAME = "facebook/nllb-200-distilled-600M"

    print(f"Loading tokenizer for model: {MODEL_NAME}")
    # For NLLB, you MUST specify the source and target languages.
    # The language codes can be found on the model card on Hugging Face Hub.
    # Luganda: "lug_Latn" (Luganda in Latin script)
    # English: "eng_Latn" (English in Latin script)
    tokenizer = NllbTokenizer.from_pretrained(
        MODEL_NAME,
        src_lang="lug_Latn",
        tgt_lang="eng_Latn"
    )

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # --- 2. Load and Prepare the Dataset ---
    print("Loading and preparing data...")
    raw_dataset = load_data(args.luganda_file, args.english_file)
    
    # We need a function to tokenize the text.
    # This will be applied to every example in our dataset.
    def preprocess_function(examples):
        # NLLB's tokenizer works best when you provide both the input and the target
        # text at the same time using `text_target`.
        inputs = [ex["lg"] for ex in examples["translation"]]
        targets = [ex["en"] for ex in examples["translation"]]
        
        # The tokenizer will correctly prepare the `input_ids` for the source (Luganda)
        # and the `labels` for the target (English).
        model_inputs = tokenizer(
            inputs, 
            text_target=targets, 
            max_length=128,  # Truncate long sentences to a max length
            truncation=True
        )
        return model_inputs

    # Apply the preprocessing function to the entire dataset.
    # The `batched=True` argument processes multiple sentences at once for speed.
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
    
    # Split the dataset into a training and a small evaluation set
    # 90% for training, 10% for validation.
    split_datasets = tokenized_dataset.train_test_split(train_size=0.9, seed=42)
    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]


    # --- 3. Set Up the Trainer ---
    print("Setting up the trainer...")
    
    # The DataCollator handles creating batches of data.
    # It will dynamically pad sentences to the same length within a batch.
    # This is more efficient than padding all sentences to the global max length.
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # These are the training arguments. They control everything about the training process.
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,          # Directory to save the model
        evaluation_strategy="epoch",         # Evaluate at the end of each epoch
        learning_rate=args.learning_rate,    # The learning rate for the optimizer
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,                  # Only keep the last 3 checkpoints
        num_train_epochs=args.epochs,
        predict_with_generate=True,          # This is required for seq2seq models
        fp16=True,                           # Use 16-bit precision for faster training on GPUs
        logging_steps=100,                   # Log progress every 100 steps
    )

    # The Seq2SeqTrainer is the main class from Hugging Face that orchestrates training.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 4. Start Training ---
    print("Starting training...")
    trainer.train()

    # --- 5. Save the Final Model ---
    print("Training complete. Saving model.")
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an NLLB model for Luganda-English translation.")
    
    # Arguments for file paths
    parser.add_argument("--luganda_file", type=str, required=True, help="Path to the Luganda text file.")
    parser.add_argument("--english_file", type=str, required=True, help="Path to the English text file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    
    # Arguments for training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")

    args = parser.parse_args()
    main(args)