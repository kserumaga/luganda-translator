# Luganda-English Translator

A project to build a high-quality, open-source translation model for Luganda to English, designed to address the lack of reliable digital resources for the language.

**Status:** 🚧 In Development / Proof-of-Concept Stage 🚧

## The Problem

Luganda is a vibrant language spoken by millions in Uganda. Despite its wide use, existing digital translation tools are often inaccurate, inconsistent, or of poor quality. This creates a barrier for communication, learning, and the development of digital tools for the Luganda-speaking community.

## Our Goal

This project aims to create a reliable and nuanced machine translation model by fine-tuning a state-of-the-art, pre-trained multilingual model on a carefully curated parallel corpus. The ultimate goal is to provide a translation service that can be integrated into applications, including the one I am currently developing.

## 🧠 Tech & Approach

This project will leverage modern tools for Natural Language Processing (NLP):

* **Model:** Fine-tuning Meta AI's **NLLB (No Language Left Behind)** model, which is designed for low-resource languages.  
* **Framework:** **Hugging Face `transformers`** for model training and management.  
* **Language:** Python  
* **Training Environment:** **Google Colab** for free GPU access.  
* **Development Environment:** VS Code for local code editing and management.  
* **Version Control:** Git & GitHub.

## 📚 Data Sources

The quality of our model depends entirely on the quality of our training data. The initial parallel corpus is being built from:

* **Primary Source:** A full parallel text of the New King James (NKJ) Bible in both Luganda and English, aligned verse-by-verse.  
* **Future Sources:** We plan to expand the dataset with news articles, children's books, and crowdsourced translations to cover more conversational and modern language.

## 🚀 Getting Started & Workflow

This project uses a hybrid local/cloud workflow to combine the benefits of a local IDE with cloud-based GPU power.

1. **Clone the Repository:**  
     
   git clone https://github.com/kserumaga/luganda-translator.git  
     
   cd luganda-translator  
     
2. **Data Setup (Not in Git):** The training data (`.txt` files) is too large for this repository and is managed via Google Drive. To replicate the training, you will need to:  
     
   * Create a folder in your Google Drive (e.g., `ML_Projects/luganda_translator/data`).  
   * Place your parallel `luganda.txt` and `english.txt` files inside it.

   

3. **Training:** The training is executed in a Google Colab notebook which:  
     
   * Clones this repository.  
   * Mounts the associated Google Drive to access the data.  
   * Installs dependencies from `requirements.txt`.  
   * Runs the training script (`train.py`).  
   * Saves the final trained model back to Google Drive.

## Roadmap

- [x] **Phase 1: Initial Data Preparation**  
      - [x] Extract and align the Luganda-English Bible corpus.  
- [ ] **Phase 2: MVP Model Training**  
      - [ ] Set up Colab environment.  
      - [ ] Write a fine-tuning script for NLLB.  
      - [ ] Train the first version of the model on the Bible corpus.  
- [ ] **Phase 3: Data Expansion & Iteration**  
      - [ ] Set up a system for crowdsourcing new translation pairs.  
      - [ ] Integrate new data sources.  
      - [ ] Re-train the model to improve its general-purpose capabilities.  
- [ ] **Phase 4: Deployment**  
      - [ ] Host the model on an inference API (e.g., Hugging Face Hub).  
      - [ ] Integrate the API into the target application.  
- [ ] **Phase 5: Speech-to-Text (ASR)**  
      - [ ] Fine-tune a speech recognition model (e.g., Whisper) on Luganda audio.

## Contributing

This is currently a solo project, but I am open to collaboration, ideas, and contributions. If you are interested in helping, especially with collecting or validating Luganda text, please open an issue to start a discussion.

## License

This project will be licensed under the MIT License. (You should create a `LICENSE` file in your repository with the full text of the MIT License).