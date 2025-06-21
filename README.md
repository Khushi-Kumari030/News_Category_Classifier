# News Category Classifier

This is a machine learning project that classifies news articles into predefined categories using Natural Language Processing (NLP). It uses TF-IDF features, NLTK-based text preprocessing, and a Logistic Regression model wrapped in a scikit-learn pipeline. A Streamlit app is built to interact with the model.

---

## Dataset

- **Source**: [BBC News Category Dataset](https://www.kaggle.com/datasets/moazeldsokyx/bbc-news)
- **Description**:  
  The dataset contains two columns:
  - `text`: News article content  
  - `category`: 5 Unique news categories `business`, `tech`, `sport`, `entertainment`, and `politics`.

---

## Data Preprocessing

Preprocessing steps (using NLTK):

- Converted text to lowercase
- Removed URLs and HTML tags
- Removed non-alphabetic characters
- Tokenized text (`nltk.word_tokenize`)
- Removed stopwords (`nltk.corpus.stopwords`)
- Applied lemmatization (`WordNetLemmatizer`)
- Identified most frequent words for each category using WordCloud

> This logic is implemented as a custom transformer and used in a scikit-learn pipeline via `FunctionTransformer` + `ColumnTransformer`.

---

## Model Pipeline

- **Preprocessing**: Custom `FunctionTransformer` wrapping your NLTK logic  
- **Vectorization**: `TfidfVectorizer` (max 5000 features)  
- **Classifier**: `LogisticRegression`  
- **Label Encoding**: `LabelEncoder` to encode/decode category labels  
- **Pipeline**: All steps combined in a `Pipeline` and saved as `news_pipeline.pkl`

---

## Model Evaluation

Model trained on an 80/20 train-test split using:
- Accuracy
- Precision, Recall, and F1-score (via `classification_report`)
- Achieved an accuracy of 0.968

---

## Streamlit App

A simple UI using Streamlit to classify user-entered news text.

### To run the app:

```bash
streamlit run app.py
