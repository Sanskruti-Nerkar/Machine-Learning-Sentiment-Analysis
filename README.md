# Sentiment Analysis using Machine Learning and Deep Learning Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


## 🎯 Overview

This project presents a comprehensive comparative analysis of **Machine Learning (ML)** and **Deep Learning (DL)** approaches for sentiment classification on textual data. The study evaluates nine different models across two distinct datasets, providing insights into model performance, generalization capabilities, and practical applicability in real-world scenarios.

**Key Highlights:**
- Comparative analysis of 9 models (6 ML + 3 DL)
- Two distinct datasets: formal movie reviews and informal e-commerce reviews
- Complete preprocessing pipeline with TF-IDF vectorization
- Deep learning models with tokenization and sequence padding
- Unsupervised clustering for sentiment pattern discovery
- Comprehensive performance evaluation with multiple metrics

## 🎯 Project Objectives

1. **Compare Traditional ML vs. Deep Learning**: Evaluate the effectiveness of classical machine learning models against modern deep learning architectures
2. **Multi-Dataset Analysis**: Test model generalization across different text domains (formal reviews vs. informal customer feedback)
3. **Performance Benchmarking**: Establish baseline metrics for sentiment classification tasks
4. **Clustering Analysis**: Discover inherent sentiment patterns using unsupervised learning techniques
5. **Practical Implementation**: Provide working code for real-world sentiment prediction

## 📊 Datasets

### Dataset 1: Rotten Tomatoes Movie Reviews

**Source**: [Hugging Face - Rotten Tomatoes Dataset](https://huggingface.co/datasets/rotten_tomatoes)

**Dataset Characteristics:**
- **Size**: 10,662 movie reviews
- **Format**: CSV with 2 columns
  - `text`: User-written movie review
  - `label`: Binary sentiment (0 = negative, 1 = positive)
- **Class Distribution**: Balanced (~50% positive, ~50% negative)
- **Text Style**: Formal, structured, professional movie critique language

**Preprocessing Applied:**
- Text cleaning (remove extra spaces, newlines, special characters)
- Lowercasing for uniformity
- HTML tag removal
- Emoji and punctuation stripping
- Tokenization using NLTK
- Stopword removal
- Lemmatization for word normalization
- TF-IDF vectorization (max_features=5000)

### Dataset 2: Flipkart Product Customer Reviews

**Source**: [Kaggle - Flipkart Product Reviews](https://www.kaggle.com/)

**Dataset Characteristics:**
- **Size**: 20,000+ customer reviews
- **Format**: CSV with 6 columns
  - `productname`: Product identifier
  - `reviewtitle`: Review headline
  - `reviewtext`: Full review content
  - `reviewrating`: Numerical rating (1-5 stars)
- **Text Style**: Informal, conversational, diverse language patterns

**Preprocessing Steps:**
1. **Rating to Sentiment Mapping:**
   - Ratings 4-5 → Positive (1)
   - Ratings 1-2 → Negative (0)
   - Rating 3 (Neutral) → Removed for binary classification

2. **Text Cleaning:**
   - Merged `reviewtitle` and `reviewtext` into single column
   - Removed duplicate reviews
   - Eliminated null/empty entries
   - Cleaned HTML tags, special characters, emojis
   - Lowercasing and stopword removal
   - Lemmatization

3. **Feature Engineering:**
   - TF-IDF vectorization (max_features=5000)
   - Train-test split: 80-20

### Combined Dataset Analysis

For comprehensive model evaluation, both datasets were merged to create a combined corpus:
- **File**: `cleaned combined 1.csv` and `cleaned_combined_data 2.csv`
- **Column Structure**: `cleaned_Review`, `Sentiment`
- **Purpose**: Test model robustness across different text styles and domains

<img width="800" height="380" alt="image" src="https://github.com/user-attachments/assets/312d8779-c503-4cd7-8824-43fdcb237424" />


<p align="center">
  <b>
Figure 2: Snippet of the Dataset</b>
</p>




## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
Jupyter Notebook (optional)
```

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/sentiment-analysis-ml-dl.git
cd sentiment-analysis-ml-dl
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
```txt
# Core Libraries
numpy>=1.19.5
pandas>=1.3.0
matplotlib>=3.4.2
seaborn>=0.11.2

# Machine Learning
scikit-learn>=0.24.2
xgboost>=1.4.2

# Deep Learning
tensorflow>=2.6.0
keras>=2.6.0

# NLP Processing
nltk>=3.6.2
transformers>=4.11.0
datasets>=1.11.0

# Clustering & Visualization
scipy>=1.7.0
plotly>=5.3.0

# Utilities
tqdm>=4.62.0
```

### Step 4: Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## 🔧 Data Preprocessing Pipeline

The preprocessing pipeline consists of multiple stages to ensure clean, normalized text data suitable for both ML and DL models.

### For Machine Learning Models (TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

**TF-IDF Parameters:**
- `max_features=5000`: Top 5000 most important words
- Converts text to numerical feature vectors
- Captures word importance using term frequency-inverse document frequency

### For Deep Learning Models (Tokenization & Padding)

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenization
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences to fixed length
maxlen = 100
X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen, 
                                padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=maxlen, 
                               padding='post', truncating='post')
```

**Tokenization Parameters:**
- `max_words=5000`: Vocabulary size
- `maxlen=100`: Maximum sequence length
- `oov_token="<OOV>"`: Out-of-vocabulary token handling
<p align="center">
<img width="600" height="392" alt="image" src="https://github.com/user-attachments/assets/6f3bb8ce-b9db-4096-9236-5b2165b49cf7" />

</p>


<p align="center">
  <b>
Figure 2: Figure of preprocessing workflow</b>
</p>


## 🤖 Models Implemented

### Machine Learning Models

#### 1. Naive Bayes
**Algorithm**: Multinomial Naive Bayes  
**Best For**: Baseline comparison, probabilistic text classification  
**Implementation:**
```python
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
```

**Characteristics:**
- Fast training and prediction
- Works well with clear polarity words
- Assumes feature independence

#### 2. Logistic Regression
**Algorithm**: Linear classification with regularization  
**Best For**: Binary sentiment classification  
**Implementation:**
```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
```

**Characteristics:**
- Balanced precision and recall
- L2 regularization to prevent overfitting
- Effective for linearly separable features

#### 3. Support Vector Machine (SVM)
**Algorithm**: Linear kernel SVM  
**Best For**: High-dimensional text data  
**Implementation:**
```python
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)
```

**Characteristics:**
- Finds optimal decision boundary
- Robust to overfitting in high dimensions
- Captures sentiment boundaries effectively

#### 4. Random Forest
**Algorithm**: Ensemble of decision trees  
**Best For**: Handling non-linear relationships  
**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)
```

**Characteristics:**
- Reduces overfitting through ensemble approach
- Handles feature interactions
- Robust to outliers

#### 5. Gradient Boosting
**Algorithm**: Sequential ensemble learning  
**Best For**: Reducing bias, improving accuracy  
**Implementation:**
```python
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                      max_depth=3, random_state=42)
gb_model.fit(X_train_tfidf, y_train)
y_pred_gb = gb_model.predict(X_test_tfidf)
```

**Characteristics:**
- Learns from previous model errors
- Gradual improvement through boosting
- Strong classification performance

#### 6. XGBoost
**Algorithm**: Optimized gradient boosting  
**Best For**: Competition-grade performance  
**Implementation:**
```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                          random_state=42)
xgb_model.fit(X_train_tfidf, y_train)
y_pred_xgb = xgb_model.predict(X_test_tfidf)
```

**Characteristics:**
- Efficient feature weighting
- Built-in regularization
- Parallel processing capabilities

### Deep Learning Models

#### 7. LSTM (Long Short-Term Memory)
**Architecture**: Recurrent neural network with memory cells  
**Best For**: Sequential text dependencies  
**Implementation:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

embedding_dim = 16
lstm_model = Sequential([
    Embedding(max_words, embedding_dim, input_length=maxlen),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', 
                   metrics=['accuracy'])
lstm_model.fit(X_train_padded, y_train, epochs=10, 
               validation_data=(X_test_padded, y_test))
```

**Characteristics:**
- Captures long-range dependencies
- Handles variable-length sequences
- Memory cells prevent vanishing gradients

#### 8. GRU (Gated Recurrent Unit)
**Architecture**: Simplified LSTM variant  
**Best For**: Faster training with comparable performance  
**Implementation:**
```python
from tensorflow.keras.layers import GRU

gru_model = Sequential([
    Embedding(max_words, embedding_dim, input_length=maxlen),
    GRU(32),
    Dense(1, activation='sigmoid')
])

gru_model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
gru_model.fit(X_train_padded, y_train, epochs=10)
```

**Characteristics:**
- Fewer parameters than LSTM
- Faster convergence
- Comparable accuracy with reduced complexity

#### 9. CNN (Convolutional Neural Network)
**Architecture**: 1D convolutions for text  
**Best For**: Local pattern and phrase detection  
**Implementation:**
```python
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dropout

cnn_model = Sequential([
    Embedding(max_words, embedding_dim, input_length=maxlen),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
cnn_model.fit(X_train_padded, y_train, epochs=10)
```

**Characteristics:**
- Captures key sentiment phrases
- Parallel processing of text segments
- Effective feature extraction through convolutions

### Unsupervised Learning Models

#### K-Means Clustering
**Purpose**: Discover natural sentiment groupings  
**Implementation:**
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_test_tfidf)
```

#### Hierarchical Clustering
**Purpose**: Visualize sentiment hierarchies  
**Implementation:**
```python
from sklearn.cluster import AgglomerativeClustering

hierarchical_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
clusters = hierarchical_model.fit_predict(X_test_tfidf.toarray())
```
<img width="700" height="350" alt="image" src="https://github.com/user-attachments/assets/95d2f63c-0c75-4861-9b4e-b84d083edc49" />
<img width="700" height="390" alt="image" src="https://github.com/user-attachments/assets/4b77a81c-7d7f-4eea-92bb-d14d846a3b5f" />



<p align="center">
  <b>
Figure 3 and 4: Model Comparison</b>
</p>



## 📈 Experimental Results

### Dataset 1: Rotten Tomatoes Movie Reviews

| Model | Accuracy | F1 Score | Precision | Recall | Training Time |
|-------|----------|----------|-----------|--------|---------------|
| **Naive Bayes** | 0.7674 | 0.7652 | 0.7680 | 0.7625 | ~2s |
| **SVM** | 0.7552 | 0.7545 | 0.7560 | 0.7530 | ~15s |
| **Logistic Regression** | 0.7495 | 0.7474 | 0.7490 | 0.7458 | ~5s |
| **Random Forest** | 0.7223 | 0.7075 | 0.7150 | 0.7001 | ~25s |
| **CNN (1D)** | 0.7195 | 0.7117 | 0.7180 | 0.7055 | ~45s |
| **XGBoost** | 0.6951 | 0.6701 | 0.6820 | 0.6585 | ~30s |
| **Gradient Boosting** | 0.6388 | 0.6816 | 0.6650 | 0.6985 | ~40s |
| **LSTM** | 0.5000 | 0.6667 | 0.5000 | 1.0000 | ~120s |
| **GRU** | 0.5000 | 0.6667 | 0.5000 | 1.0000 | ~100s |


**Key Observations:**
- Naive Bayes achieved highest accuracy (76.74%) on structured movie reviews
- LSTM and GRU struggled, likely due to insufficient data for deep learning
- Traditional ML models outperformed deep learning on this smaller dataset
- SVM and Logistic Regression showed balanced performance

### Dataset 2: Flipkart Product Reviews

| Model | Accuracy | F1 Score (Weighted) | Training Time |
|-------|----------|---------------------|---------------|
| **CNN** | 0.9038 | 0.9035 | ~60s |
| **Logistic Regression** | 0.9035 | 0.9032 | ~8s |
| **Random Forest** | 0.9035 | 0.9030 | ~35s |
| **XGBoost** | 0.9035 | 0.9033 | ~40s |
| **SVM** | 0.9033 | 0.9030 | ~20s |
| **Gradient Boosting** | 0.9031 | 0.9028 | ~50s |
| **Naive Bayes** | 0.8960 | 0.8955 | ~3s |
| **LSTM** | 0.8006 | 0.7995 | ~150s |
| **GRU** | 0.8006 | 0.7990 | ~130s |


**Key Observations:**
- CNN achieved highest accuracy (90.38%) on larger e-commerce dataset
- Multiple models clustered around 90% accuracy
- Deep learning models showed improvement with larger dataset
- All models performed better than on Dataset 1 due to data size

### Combined Dataset Performance

| Model | Accuracy | Remarks |
|-------|----------|---------|
| **XGBoost** | 92.1% | Best overall performance with efficient feature weighting |
| **Random Forest** | 90.2% | Robust handling of mixed features |
| **CNN** | 89.7% | Effective capture of sentiment phrases |
| **Gradient Boosting** | 88.6% | Strong classification with reduced bias |
| **GRU** | 86.2% | Faster convergence than LSTM |
| **LSTM** | 85.4% | Good sequential dependency learning |
| **SVM** | 77.1% | Better boundary capture |
| **Logistic Regression** | 75.8% | Balanced precision and recall |
| **Naive Bayes** | 74.3% | Good with clear polarity words |

<img width="800" height="490" alt="image" src="https://github.com/user-attachments/assets/a95eb88c-ea48-40a8-aaad-a6fd52b5757c" />
<p align="center">
  <b>
Figure 5: Visualization of Performance Metrics</b>
</p>


### Performance Hierarchy

**XGBoost > CNN > Random Forest > GRU > LSTM > SVM > Logistic Regression > Naive Bayes**

![Performance Hierarchy](./results/plots/performance_hierarchy.png)
*Figure 7: Overall model ranking across all experiments*

## 📊 Visualization & Analysis

### Confusion Matrices

Confusion matrices provide detailed insight into model classification behavior, showing true positives, true negatives, false positives, and false negatives.

![Confusion Matrices - All Models](./results/plots/confusion_matrices_all.png)
*Figure 8: Confusion matrices for all models on Dataset 1*

**Key Insights from Confusion Matrices:**
- Most models showed higher false positives than false negatives
- Neutral sentiments (when present) were frequently misclassified as positive
- Ensemble models showed more balanced error distribution
- Deep learning models had higher variance in predictions

### Model Accuracy Comparison

![Accuracy Bar Chart](./results/plots/accuracy_comparison.png)
*Figure 9: Accuracy comparison across all models*

### F1 Score Comparison

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/59b452bf-4a35-4f26-97d4-75751fac58ba" />

*Figure 10: F1 score comparison across all models*

### Cross-Dataset Performance

![Cross-Dataset Performance](./results/plots/cross_dataset_comparison.png)
*Figure 11: Model performance comparison across datasets*

**Analysis:**
- Models showed significant performance variation across datasets
- Dataset 2 (larger, more diverse) yielded better results for all models
- Deep learning models showed larger improvement with increased data
- Traditional ML maintained consistent performance across datasets

### Training History (Deep Learning)

![LSTM Training History](./results/plots/lstm_training_history.png)
*Figure 12: LSTM training and validation accuracy over epochs*

![CNN Training History](./results/plots/cnn_training_history.png)
*Figure 13: CNN training and validation accuracy over epochs*

### Clustering Visualizations

![K-Means Clustering](./results/plots/kmeans_clusters.png)
*Figure 14: K-Means clustering of sentiment data (k=2)*

![Hierarchical Clustering Dendrogram](./results/plots/hierarchical_dendrogram.png)
*Figure 15: Hierarchical clustering dendrogram showing sentiment groupings*

**Clustering Insights:**
- K-Means successfully separated positive and negative sentiments
- Hierarchical clustering revealed sub-groups within sentiment categories
- Cluster assignments aligned well with supervised model predictions
- Unsupervised methods validated supervised learning results

### Word Clouds

![Positive Sentiment Word Cloud](./results/plots/wordcloud_positive.png)
*Figure 16: Most frequent words in positive reviews*

![Negative Sentiment Word Cloud](./results/plots/wordcloud_negative.png)
*Figure 17: Most frequent words in negative reviews*

**Word Cloud Analysis:**
- **Positive words**: "great", "excellent", "love", "amazing", "perfect", "best"
- **Negative words**: "poor", "waste", "bad", "terrible", "worst", "disappointed"
- Clear lexical distinction between sentiment classes
- Domain-specific vocabulary emerged (e.g., "plot" in movies, "product" in e-commerce)

## 💻 Usage Guide

### 1. Training Models

```python
# Load data
train_df = pd.read_csv('cleaned_train_data.csv')
test_df = pd.read_csv('cleaned_test_data.csv')

X_train = train_df['text']
y_train = train_df['label']
X_test = test_df['text']
y_test = test_df['label']

# Vectorize for ML models
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Train Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Train XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_tfidf, y_train)
```

### 2. Predicting Sentiment on New Text

```python
# New text samples
new_texts = [
    "This movie was absolutely fantastic!",
    "Waste of time and money. Terrible product."
]

# Preprocess and vectorize
new_texts_tfidf = tfidf_vectorizer.transform(new_texts)

# Predict using trained model
predictions = nb_model.predict(new_texts_tfidf)

# Display results
sentiment_map = {0: 'Negative', 1: 'Positive'}
for text, pred in zip(new_texts, predictions):
    print(f"Text: '{text}'")
    print(f"Predicted Sentiment: {sentiment_map[pred]}\n")
```

**Output:**
```
Text: 'This movie was absolutely fantastic!'
Predicted Sentiment: Positive

Text: 'Waste of time and money. Terrible product.'
Predicted Sentiment: Negative
```

### 3. Training Deep Learning Models

```python
# Prepare data for deep learning
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_words = 5000
maxlen = 100

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen, 
                                padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=maxlen, 
                               padding='post', truncating='post')

# Train LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

lstm_model = Sequential([
    Embedding(max_words, 16, input_length=maxlen),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', 
                   metrics=['accuracy'])
lstm_model.fit(X_train_padded, y_train, epochs=10, 
               validation_data=(X_test_padded, y_test))

# Predict
predictions = (lstm_model.predict(X_test_padded) > 0.5).astype("int32")
```

### 4. Model Evaluation

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get predictions
y_pred = model.predict(X_test_tfidf)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### 5. Saving and Loading Models

```python
# Save scikit-learn model
import joblib
joblib.dump(xgb_model, 'models/xgboost_sentiment.pkl')
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')

# Load model
loaded_model = joblib.load('models/xgboost_sentiment.pkl')
loaded_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Save Keras model
lstm_model.save('models/lstm_sentiment.h5')

# Load Keras model
from tensorflow.keras.models import load_model
loaded_lstm = load_model('models/lstm_sentiment.h5')
```

## 🔍 Key Findings & Insights

### 1. Dataset Impact on Performance

**Finding**: Model performance is highly dependent on dataset characteristics
- Larger datasets (20,000+ samples) significantly improved deep learning model accuracy
- Smaller datasets (<11,000 samples) favored traditional ML approaches
- Text formality and structure affected model behavior

**Implication**: Choose models based on available data volume:
- **Small datasets** (<10k): Naive Bayes, Logistic Regression, SVM
- **Medium datasets** (10k-50k): Random Forest, XGBoost, CNN
- **Large datasets** (>50k): LSTM, GRU, Transformer models

### 2. Preprocessing is Critical

**Finding**: Data preprocessing significantly impacted all model accuracies
- Proper tokenization improved accuracy by 5-10%
- Stopword removal and lemmatization enhanced model generalization
- TF-IDF vectorization was crucial for ML model performance

**Key Steps**:
1. Text cleaning (remove HTML, special characters)
2. Lowercasing
3. Stopword removal
4. Lemmatization
5. Appropriate vectorization (TF-IDF vs. tokenization)

### 3. Ensemble Models Dominate

**Finding**: Ensemble methods (XGBoost, Random Forest) consistently outperformed single models
- XGBoost achieved 92.1% accuracy on combined dataset
- Random Forest showed robust performance across different text styles
- Gradient Boosting demonstrated strong bias reduction

**Advantage**: Ensemble models combine multiple weak learners, reducing overfitting and improving generalization

### 4. Deep Learning Requires Scale

**Finding**: Deep learning models (LSTM, GRU, CNN) showed poor performance on small datasets
- LSTM/GRU achieved only 50% accuracy on Rotten Tomatoes (10k samples)
- Same models reached 80-90% accuracy on larger Flipkart dataset (20k+ samples)
- CNN showed best deep learning performance overall

**Recommendation**: Use deep learning only when:
- Dataset size > 20,000 samples
- Computational resources available
- Sequential dependencies important (LSTM/GRU)
- Local patterns matter (CNN)

### 5. CNN Surprises in Text Classification

**Finding**: CNN outperformed RNNs on sentiment analysis tasks
- CNN achieved 90.38% accuracy vs. 80.06% for LSTM/GRU on Dataset 2
- Faster training time (60s vs. 150s for LSTM)
- Better at capturing key sentiment phrases

**Explanation**: 
- Sentiment is often determined by local phrases ("very good", "completely terrible")
- CNNs excel at detecting these local patterns
- RNNs may be unnecessary for simple sentiment tasks

### 6. Class Imbalance Effects

**Finding**: Models tended to misclassify neutral sentiments as positive
- Confusion matrices showed bias toward positive class
- Likely due to positive class being more common in training data
- Affected precision-recall balance

**Solution**: 
- Use weighted F1 score for multi-class problems
- Consider class balancing techniques (oversampling, SMOTE)
- Adjust classification thresholds based on business needs

### 7. Unsupervised Validation

**Finding**: K-Means and Hierarchical Clustering confirmed supervised model findings
- Clear separation between positive and negative clusters
- Cluster assignments aligned with true labels
- Validated that sentiment classes are indeed separable

**Use Case**: Unsupervised methods useful for:
- Initial data exploration
- Validating labeled data quality
- Discovering sentiment sub-categories

