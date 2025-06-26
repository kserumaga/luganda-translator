# train.py

import argparse
import os
import torch
from datasets import Dataset
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
# HF MODIFICATION: Import the login function
from huggingface_hub import login

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
            if "Luganda" in df.columns and "English" in df.columns:
                lug_lines = df["Luganda"].astype(str).tolist()
                eng_lines = df["English"].astype(str).tolist()
            else:
                # Try using column C (index 2) as specified in your previous code
                print("Could not find 'Luganda' and 'English' columns. Using column C (index 2) for data extraction")
                if len(df.columns) > 2:
                    lug_lines = df.iloc[:, 2].astype(str).tolist()
                    # Assuming English is in the next column
                    eng_lines = df.iloc[:, 3].astype(str).tolist() if len(df.columns) > 3 else []
                else:
                    raise ValueError("Excel file does not have enough columns. Need column C or 'Luganda'/'English' columns.")
        else:
            # Two separate Excel files
            df_lug = pd.read_excel(luganda_excel_path)
            df_eng = pd.read_excel(english_excel_path)
            
            # Try using column C (index 2) for both files 
            if len(df_lug.columns) > 2 and len(df_eng.columns) > 2:
                print("Using column C (index 2) from both Excel files")
                lug_lines = df_lug.iloc[:, 2].astype(str).tolist()
                eng_lines = df_eng.iloc[:, 2].astype(str).tolist()
            else:
                # Fall back to the first column if column C doesn't exist
                print("Using first column from both Excel files")
                lug_lines = df_lug.iloc[:, 0].astype(str).tolist()
                eng_lines = df_eng.iloc[:, 0].astype(str).tolist()
        
        print(f"Loaded {len(lug_lines)} Luganda lines and {len(eng_lines)} English lines")
        
        # Filter out 'nan' strings and empty strings
        data = {"translation": []}
        for i in range(min(len(lug_lines), len(eng_lines))):
            lg, en = str(lug_lines[i]).strip(), str(eng_lines[i]).strip()
            if lg and en and lg.lower() != 'nan' and en.lower() != 'nan':
                data["translation"].append({"en": en, "lg": lg})
        
        print(f"Created dataset with {len(data['translation'])} valid translation pairs")
        
        # Display a few samples to verify the data
        sample_size = min(3, len(data["translation"]))
        print("\nData samples:")
        for i in range(sample_size):
            print(f"Sample {i+1}:")
            print(f"  Luganda: {data['translation'][i]['lg'][:50]}...")
            print(f"  English: {data['translation'][i]['en'][:50]}...")
        
        return Dataset.from_dict(data)
    except Exception as e:
        print(f"Error loading Excel files: {e}")
        raise

def compute_metrics(eval_preds):
    """Compute evaluation metrics for the model."""
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace padding token id with tokenizer pad id
    labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode the predictions and references
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Sample a few predictions to show in the logs
    result = {"samples": []}
    num_samples = min(3, len(decoded_preds))
    for i in range(num_samples):
        result["samples"].append({
            "reference": decoded_labels[i],
            "prediction": decoded_preds[i]
        })
    
    return result

def main(args):
    # HF MODIFICATION: Login to the Hugging Face Hub using the token provided as a secret.
    print("Logging into Hugging Face Hub...")
    if args.hf_token:
        login(token=args.hf_token)
    else:
        print("No HF token provided, proceeding without login (will fail if push_to_hub=True)")
    
    # --- 1. Load Model and Tokenizer ---
    MODEL_NAME = "facebook/nllb-200-distilled-600M"
    print(f"Loading tokenizer for model: {MODEL_NAME}")
    tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME, src_lang="lug_Latn", tgt_lang="eng_Latn")

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True, # Good for loading large models
    )

    # --- 2. Load and Prepare the Dataset ---
    print("Loading and preparing data from Excel...")
    try:
        # We assume the excel files are in the same directory as the script.
        raw_dataset = load_data_from_excel(args.luganda_excel, args.english_excel)
    except Exception as e:
        print(f"Fatal error: Failed to load data. Exiting. Error: {e}")
        return

    def preprocess_function(examples):
        inputs = [ex["lg"] for ex in examples["translation"]]
        targets = [ex["en"] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
    split_datasets = tokenized_dataset.train_test_split(train_size=0.9, seed=42)
    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]

    # --- 3. Set Up the Trainer ---
    print("Setting up the trainer...")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Gradient accumulation logic from your original script is great for memory saving
    per_device_batch = 4
    gradient_accumulation = max(1, args.batch_size // per_device_batch)
    
    # Make sure we evaluate regularly (about 10 times per epoch)
    eval_steps = max(len(train_dataset) // (args.batch_size * 10), 1)

    # HF MODIFICATION: Configure TrainingArguments to push to the hub
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,           # Local output directory
        hub_model_id=args.hub_model_id,       # HF Hub model ID for pushing
        eval_strategy="steps",                # Evaluate regularly
        eval_steps=eval_steps,                # About 10 evaluations per epoch
        learning_rate=args.learning_rate,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=True,                            # Essential for performance on modern GPUs
        logging_steps=100,
        push_to_hub=True if args.hf_token else False,  # Only push if token is provided
        report_to="none",                     # Disable wandb reporting
        save_steps=eval_steps,                # Save at the same frequency as evaluation
        load_best_model_at_end=True,          # Load best model at the end of training
        metric_for_best_model="eval_loss",
        greater_is_better=False,              # Lower loss is better
    )

    # Add early stopping to prevent overfitting
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,            # Stop if no improvement for 3 evaluations
        early_stopping_threshold=0.01         # Minimum improvement needed
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,      # Added metrics computation
        callbacks=[early_stopping],           # Added early stopping
    )

    # --- 4. Start Training ---
    print(f"Starting training with effective batch size {args.batch_size} (device batch: {per_device_batch}, accumulation: {gradient_accumulation})")
    
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        # Try to save anyway
        print("Attempting to save partial model...")
        trainer.save_model(os.path.join(args.output_dir, "partial"))
        return

    # --- 5. Save the Final Model ---
    if args.hf_token:
        print(f"Training complete. Pushing final model to Hub: {args.hub_model_id}")
        trainer.push_to_hub()
    else:
        print(f"Training complete. Saving model locally to: {args.output_dir}")
        trainer.save_model(args.output_dir)
    
    # Save tokenizer explicitly to ensure all necessary files are saved
    tokenizer.save_pretrained(args.output_dir)
    
    # Try a test translation to verify the model works
    try:
        test_text = "Katonda"
        inputs = tokenizer(test_text, return_tensors="pt")
        outputs = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
            max_length=128
        )
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"\nTest translation - Luganda: '{test_text}' → English: '{translation}'")
    except Exception as e:
        print(f"Test translation failed: {e}")
    
    print("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an NLLB model for Luganda-English translation on Hugging Face Spaces.")
    
    # HF MODIFICATION: Updated arguments for the HF Hub workflow
    parser.add_argument("--luganda_excel", type=str, required=True, help="Filename of the Excel file with Luganda text.")
    parser.add_argument("--english_excel", type=str, default=None, help="Filename of the Excel file with English text (optional, if not in the same file).")
    
    parser.add_argument("--hub_model_id", type=str, default=None, help="Your Hugging Face Hub model ID (e.g., 'your-username/nllb-luganda-english').")
    parser.add_argument("--hf_token", type=str, default=None, help="Your Hugging Face Hub write token (passed as a secret).")
    
    # Standard hyperparameters
    parser.add_argument("--output_dir", type=str, default="./results", help="Local directory for temporary checkpoints.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Effective batch size (will be simulated with gradient accumulation).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")

    args = parser.parse_args()
    main(args)