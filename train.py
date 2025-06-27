"""
Fine-tune NLLB model for Luganda-English translation
"""

import argparse
import os
import torch
from datasets import Dataset, load_dataset
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from huggingface_hub import login
import pandas as pd
import numpy as np

def load_data_from_excel(luganda_excel_path, english_excel_path=None):
    """Load translation data directly from Excel files."""
    try:
        print(f"Loading data from Excel files...")
        
        # Check if it's a single file or two files
        if english_excel_path is None:
            # Single Excel file with both languages
            df = pd.read_excel(luganda_excel_path)
            # Try various formats
            if "Luganda" in df.columns and "English" in df.columns:
                lug_lines = df["Luganda"].astype(str).tolist()
                eng_lines = df["English"].astype(str).tolist()
                print("Using 'Luganda' and 'English' named columns")
            elif len(df.columns) > 2:
                # Use column C for Luganda (index 2) and D for English (index 3)
                print("Using column C for Luganda and D for English")
                lug_lines = df.iloc[:, 2].astype(str).tolist()
                eng_lines = df.iloc[:, 3].astype(str).tolist() if len(df.columns) > 3 else []
            else:
                # Use first two columns
                print("Using first two columns for Luganda and English")
                lug_lines = df.iloc[:, 0].astype(str).tolist()
                eng_lines = df.iloc[:, 1].astype(str).tolist() if len(df.columns) > 1 else []
        else:
            # Two separate Excel files
            df_lug = pd.read_excel(luganda_excel_path)
            df_eng = pd.read_excel(english_excel_path)
            
            print(f"Loaded Luganda Excel with {len(df_lug)} rows")
            print(f"Loaded English Excel with {len(df_eng)} rows")
            
            # Try to use column C (index 2) if available
            if len(df_lug.columns) > 2 and len(df_eng.columns) > 2:
                lug_lines = df_lug.iloc[:, 2].astype(str).tolist()
                eng_lines = df_eng.iloc[:, 2].astype(str).tolist()
                print("Using column C from both Excel files")
            else:
                # Use first column
                lug_lines = df_lug.iloc[:, 0].astype(str).tolist()
                eng_lines = df_eng.iloc[:, 0].astype(str).tolist()
                print("Using first column from both Excel files")
        
        print(f"Loaded {len(lug_lines)} Luganda lines and {len(eng_lines)} English lines")
        
        # Filter out nan values and empty strings
        data = {"translation": []}
        for i in range(min(len(lug_lines), len(eng_lines))):
            lg, en = str(lug_lines[i]).strip(), str(eng_lines[i]).strip()
            if lg and en and lg.lower() != 'nan' and en.lower() != 'nan':
                data["translation"].append({"en": en, "lg": lg})
        
        print(f"Created dataset with {len(data['translation'])} valid translation pairs")
        
        # Sample a few examples
        for i in range(min(3, len(data["translation"]))):
            print(f"Sample {i+1}:")
            print(f"  Luganda: {data['translation'][i]['lg']}")
            print(f"  English: {data['translation'][i]['en']}")
            
        return Dataset.from_dict(data)
    except Exception as e:
        print(f"Error loading Excel files: {e}")
        raise

def load_data_from_hub(dataset_name, split="train"):
    """Load translation data from Hugging Face dataset."""
    try:
        print(f"Loading data from Hugging Face dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        
        # Check for different dataset formats
        if "translation" in dataset.column_names:
            # Dataset has a translation field (like OPUS datasets)
            if isinstance(dataset[0]["translation"], dict) and "en" in dataset[0]["translation"] and "lg" in dataset[0]["translation"]:
                # Convert to our format
                data = {"translation": []}
                for item in dataset:
                    data["translation"].append({
                        "en": item["translation"]["en"],
                        "lg": item["translation"]["lg"]
                    })
                return Dataset.from_dict(data)
        
        # Direct column format (en, lg columns)
        if "en" in dataset.column_names and "lg" in dataset.column_names:
            data = {"translation": []}
            for item in dataset:
                data["translation"].append({
                    "en": item["en"],
                    "lg": item["lg"]
                })
            
            print(f"Created dataset with {len(data['translation'])} valid translation pairs")
            
            # Sample a few examples
            for i in range(min(3, len(data["translation"]))):
                print(f"Sample {i+1}:")
                print(f"  Luganda: {data['translation'][i]['lg']}")
                print(f"  English: {data['translation'][i]['en']}")
                
            return Dataset.from_dict(data)
        
        # If we reach here, we couldn't find the right format
        print(f"Warning: Unexpected dataset format. Found columns: {dataset.column_names}")
        print(f"First example: {dataset[0]}")
        raise ValueError("Dataset does not have the expected column format")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        raise

def compute_metrics(eval_preds, tokenizer):
    """Compute evaluation metrics for the model."""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Simple metrics for monitoring
    result = {"sample_predictions": []}
    
    # Add some sample predictions for monitoring
    num_samples = min(3, len(decoded_preds))
    for i in range(num_samples):
        result["sample_predictions"].append({
            "prediction": decoded_preds[i],
            "reference": decoded_labels[i]
        })
        
    return result

def main(args):
    # HF MODIFICATION: Login to the Hugging Face Hub
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
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # --- 2. Load and Prepare the Dataset ---
    print("Loading and preparing data...")
    
    try:
        if args.dataset:
            # Load from Hugging Face dataset
            raw_dataset = load_data_from_hub(args.dataset)
        elif args.luganda_excel:
            # Load from Excel files
            raw_dataset = load_data_from_excel(args.luganda_excel, args.english_excel)
        else:
            raise ValueError("Either --dataset or --luganda_excel must be provided")
    except Exception as e:
        print(f"Fatal error: Failed to load data. Exiting. Error: {e}")
        return

    # We need a function to tokenize the text
    def preprocess_function(examples):
        inputs = [ex["lg"] for ex in examples["translation"]]
        targets = [ex["en"] for ex in examples["translation"]]
        
        model_inputs = tokenizer(
            inputs, 
            text_target=targets, 
            max_length=128,
            truncation=True
        )
        return model_inputs

    # Apply the preprocessing function to the entire dataset
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
    
    # Split the dataset into a training and a small evaluation set
    split_datasets = tokenized_dataset.train_test_split(train_size=0.9, seed=42)
    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]

    # --- 3. Set Up the Trainer ---
    print("Setting up the trainer...")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Optimize batch size based on available hardware
    per_device_batch = args.batch_size
    gradient_accumulation = 4  # Accumulate gradients over 4 steps

    # Calculate ≈ 10 eval steps per epoch to get frequent feedback
    eval_steps = max(len(train_dataset) // (per_device_batch * gradient_accumulation * 10), 1)
    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id if args.hub_model_id else None,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation,
        weight_decay=0.01,
        save_total_limit=3, 
        save_steps=eval_steps,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=True if torch.cuda.is_available() else False,
        logging_steps=100,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=args.hub_model_id is not None,
        optim="adamw_torch",
    )

    # Create a metric calculation function
    def compute_metrics_wrapper(eval_preds):
        return compute_metrics(eval_preds, tokenizer)

    # Add early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # Stop if no improvement for 3 evaluations
        early_stopping_threshold=0.01  # Minimum improvement needed
    )

    # Set up the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
        callbacks=[early_stopping_callback],
    )

    # --- 4. Start Training ---
    effective_batch = per_device_batch * gradient_accumulation
    print(f"Starting training with effective batch size {effective_batch} (device batch: {per_device_batch}, gradient accumulation: {gradient_accumulation})")
    
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        # Try to save the model anyway
        print("Attempting to save the partial model...")
        trainer.save_model(os.path.join(args.output_dir, "partial_model"))
        print("Partial model saved.")
        return

    # --- 5. Save the Final Model ---
    print("Training complete. Saving model.")
    
    model_save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print(f"Model and tokenizer saved to {model_save_path}")

    # Create a simple test to verify the model works
    test_luganda = "Katonda"
    inputs = tokenizer(test_luganda, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
        max_length=128
    )
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Test translation - Luganda: '{test_luganda}' → English: '{translation}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an NLLB model for Luganda-English translation.")
    
    # Data source options (either dataset or Excel files)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset", type=str, help="Hugging Face dataset name (e.g., 'kserumaga/luganda-english-bible')")
    data_group.add_argument("--luganda_excel", type=str, help="Path to Excel file with Luganda text.")
    
    # Excel-specific option
    parser.add_argument("--english_excel", type=str, default=None, help="Path to Excel file with English text (optional if in the same file).")
    
    # Hugging Face Hub options
    parser.add_argument("--hub_model_id", type=str, default=None, help="Your Hugging Face Hub model ID (e.g., 'kserumaga/luganda-nllb').")
    parser.add_argument("--hf_token", type=str, default=None, help="Your Hugging Face Hub write token.")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the fine-tuned model.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training per device.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")

    args = parser.parse_args()
    main(args)