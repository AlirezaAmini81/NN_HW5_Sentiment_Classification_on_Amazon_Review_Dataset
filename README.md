# Sentiment Classification on Amazon Review Dataset

## Overview
This project aims to build a robust system for sentiment classification, specifically designed to analyze user reviews for products. The primary objective is to classify reviews as positive or negative based on textual data. By leveraging advanced recurrent neural network architectures such as **RNN**, **LSTM**, and **GRU**, the project explores their effectiveness in handling sequential data and extracting meaningful patterns for sentiment prediction.

The dataset utilized includes reviews from the Amazon Reviews Dataset (2018), focusing on industrial and scientific products. Reviews are preprocessed to tokenize text and convert them into numerical representations for model training.

## Features
- **Multi-Model Approach**: Implements and compares RNN, LSTM, and GRU architectures.
- **Dataset Preprocessing**: Tokenization and numerical encoding for seamless input to neural networks.
- **Performance Optimization**: Identifies the best-performing architecture through extensive experimentation.
- **Versatile Applications**: Applicable to broader domains requiring sentiment analysis, such as customer feedback and social media monitoring.


## Usage
### Training the Model
1. Prepare the dataset:
   - Load the Amazon Reviews Dataset or a similar dataset.
   - Tokenize and preprocess the review texts using tools like `spacy`.

2. Train the model:
   ```python
   from training import train_model
   config = {
       'model_type': 'GRU',  # Options: 'RNN', 'LSTM', 'GRU'
       'num_layers': 2,
       'optimizer': 'Adam',
       'learning_rate': 0.001
   }
   train_model(dataset, config)
   ```

### Evaluating the Model
1. Test the model:
   ```python
   from evaluation import evaluate_model
   accuracy = evaluate_model(test_dataset, trained_model)
   print(f"Model Accuracy: {accuracy}%")
   ```

## Results
### Model Comparison
- **RNN**: Achieved moderate accuracy but struggled with overfitting on smaller datasets.
- **LSTM**: Performed better than RNN but required extensive tuning for optimal results.
- **GRU**: Delivered the best performance with a 2-layer architecture.

### Best Model Performance
- **Architecture**: 2-layer GRU
- **Validation Accuracy**: 77.01%
- **Test Accuracy**: Consistent with validation results


## Technologies Used
- **Programming Language**: Python
- **Libraries**: PyTorch, NumPy, Matplotlib, Spacy
- **Dataset**: Amazon Reviews Dataset (5-core subset)

