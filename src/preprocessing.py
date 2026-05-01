from datasets import load_dataset
import re
import spacy
import unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.lda_model
from gensim.models.phrases import Phrases, Phraser
from collections import Counter
import matplotlib.pyplot as plt


def clean_text(text: str) -> str:
    """Preprocess raw text for topic modeling."""
    if not isinstance(text, str):
        return None
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[’']", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\b([a-z])\1{1,}\b", "", text)
    text = re.sub(r"\b(?!us\b|eu\b)[a-z]{1,2}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_lemmatized_corpus(df, stopwords_path="custom_stopwords_bis.txt"):
    """Tokenization, stopword removal and lemmatization."""
    # stopwords
    stop_words = set(ENGLISH_STOP_WORDS)
    with open(stopwords_path) as f:
        custom_stopwords = {line.strip() for line in f if line.strip()}
    STOPWORDS = stop_words | custom_stopwords
    # tokenize
    df["tokens"] = df["clean_text"].apply(lambda x: x.split())
    # remove stopwords
    df["tokens_clean"] = df["tokens"].apply(
        lambda tokens: [w for w in tokens if w not in STOPWORDS]
    )
    # lemmatization
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    df["tokens_lemma"] = df["tokens_clean"].apply(
        lambda tokens: [token.lemma_ for token in nlp(" ".join(tokens))]
    )
    return df


def build_ngrams(df, min_count=5, threshold=15):
    """Construct bigram and trigram features."""
    bigram = Phrases(df["tokens_lemma"], min_count=min_count, threshold=threshold)
    trigram = Phrases(bigram[df["tokens_lemma"]], threshold=threshold)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    df["tokens_final"] = df["tokens_lemma"].apply(
        lambda tokens: trigram_mod[bigram_mod[tokens]]
    )
    return df


def build_preprocessed_corpus(
    df,
    stopwords_path="custom_stopwords_bis.txt",
    min_count=5,
    threshold=15):
    
    """Full preprocessing pipeline for topic modeling."""

    df = df[df["text"].notna()]
    df = df[df["text"].str.strip() != ""]
    df = df[df["text"].str.contains("[a-zA-Z]", na=False)]

    df["text_length"] = df["text"].str.len()
    df = df[(df["text_length"] > 500) & (df["text_length"] < 100000)]

    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].notna()]
    df = df[df["clean_text"].str.strip() != ""]

    df = build_lemmatized_corpus(df, stopwords_path=stopwords_path)
    df = build_ngrams(df, min_count=min_count, threshold=threshold)

    df["text_final"] = df["tokens_final"].apply(lambda tokens: " ".join(tokens))
    df = df[df["text_final"].str.strip() != ""]

    return df