# CreditFlow AI: Loan Defaulter Prediction

## Overview 💑

This project focuses on predicting loan-defaulters using machine learning techniques. This model uses historical loan data to predict the likelihood of default for new loan applications. It incorporates various borrower characteristics and loan attributes to generate risk scores and support credit underwriting decisions. The model leverages **TensorFlow** and **Scikit-Learn** for training, along with data preprocessing and visualization libraries such as **Pandas, Seaborn, and Matplotlib**.

## Features 📂

- **Comprehensive Data Preprocessing**: Handling missing values, feature scaling, and categorical encoding to ensure high-quality data input.
- **Feature Engineering**: Creating new meaningful features that improve model performance, including debt-to-income ratio, credit score categorization, and loan history aggregation.
- **Neural Network Model Training**: Implemented using TensorFlow/Keras with optimized hyperparameters for better accuracy and generalization.
- **Performance Evaluation**: Utilizing accuracy, precision-recall, F1-score, and ROC-AUC metrics for an in-depth analysis of model effectiveness.
- **Result Visualization**: Graphical representation of performance via ROC curve, confusion matrix, feature importance plots, and trend analysis.

## Requirements ⚙️

Ensure you have the following dependencies installed:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow hvplot
```

## Installation 💾

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/NK-Works/CreditFlow-AI
cd CreditFlow-AI
```

## Dataset 📊

The dataset used in this project contains loan applicant information, with labeled instances of whether the applicant defaulted. Ensure your dataset is available in the correct path before running the notebook.

You can download the dataset from [here](https://drive.google.com/file/d/1837s3zKxAIlWEACd7iqzqeNjDNVwlspl/view?usp=sharing)

## Usage 🚀

Run the Jupyter Notebook to preprocess data and train the model:

```bash
jupyter notebook model.ipynb
```

Alternatively, execute it in Google Colab by uploading the notebook and dataset.

## Model Architecture 🏠

The model is implemented using TensorFlow/Keras and follows an **Artificial Neural Network (ANN)** architecture designed for optimal performance in loan default prediction. The key layers include:

- **Fully Connected Dense Layers**: Multiple dense layers with **ReLU activation** function to enable non-linearity and better representation learning.
- **Dropout Layers**: Applied after dense layers to prevent overfitting by randomly disabling neurons during training.
- **Batch Normalization**: Ensures training stability by normalizing inputs at each layer, accelerating convergence.
- **Adam Optimizer**: Adaptive learning rate optimization that improves training efficiency and prevents vanishing gradients.
- **Output Layer**: A final layer using the **sigmoid activation function** to generate probability scores for loan default predictions.

This architecture helps the model generalize well to unseen data, contributing to its high accuracy of over 88%.

## Evaluation Metrics 📈

- **Accuracy**: Measures overall correctness of the model
- **Confusion Matrix**: Visualizes classification performance
- **ROC-AUC Score**: Assesses model’s ability to distinguish between classes

## Results 🏆

The trained model has achieved an accuracy of over **88%**, making it highly reliable for predicting loan defaulters. It is evaluated using multiple performance metrics and visualized through:

- **ROC Curve**: Demonstrates the true positive rate versus the false positive rate, showing strong discriminatory power.
- **Confusion Matrix**: Provides an overview of correct and incorrect classifications, highlighting the model's precision and recall.
- **Feature Importance Analysis**: Identifies key factors influencing loan default predictions, helping financial institutions make informed decisions.

## License 📚

This project is licensed under the MIT License.

**Credits:** This project was developed under the guidance of **Infosys**.

---
