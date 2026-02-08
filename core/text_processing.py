import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")

class TextProcessor:
    def __init__(self, df):
        self.df = df
        self.stop_words = set(stopwords.words("english"))

    # -----------------------------
    # Basic text cleaning
    # -----------------------------
    def basic_clean(self, col):
        self.df[col] = self.df[col].apply(self._safe_clean)
        return self.df

    def _safe_clean(self, value):
        if pd.isna(value):
            return ""

        text = str(value).lower()
        text = "".join(c for c in text if c not in string.punctuation)
        text = " ".join(w for w in text.split() if w not in self.stop_words)
        return text

    # -----------------------------
    # Length-based features
    # -----------------------------
    def add_length_features(self, col):
        self.df[f"{col}_char_count"] = self.df[col].apply(self._safe_char_len)
        self.df[f"{col}_word_count"] = self.df[col].apply(self._safe_word_len)
        return self.df

    def _safe_char_len(self, value):
        if pd.isna(value):
            return 0
        return len(str(value))

    def _safe_word_len(self, value):
        if pd.isna(value):
            return 0
        return len(str(value).split())

    # -----------------------------
    # TF-IDF features
    # -----------------------------
    def tfidf_features(self, col, max_features=10):
        # Replace NaN with empty string
        text_series = self.df[col].fillna("").astype(str)

        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(text_series)

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{w}" for w in vectorizer.get_feature_names_out()]
        )

        self.df = pd.concat(
            [self.df.reset_index(drop=True), tfidf_df],
            axis=1
        )

        return self.df
