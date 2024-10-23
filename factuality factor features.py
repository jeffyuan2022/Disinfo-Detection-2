# -*- coding: utf-8 -*-
"""dsc180A_factor feature engineering.ipynb Jade Oct.14

Factuality factors: linguistic based, toxicity

Skipping the data cleanning process

Feature engineering could be directly added to Jeff's code
"""

import nltk
nltk.download('punkt', quiet=True)

from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import spacy  # For named entity recognition (NER)
import pandas as pd
import numpy as np
import re

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')  # Load the small English language model for spaCy

class SensationalismFeatureGenerator:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english', max_features=500)

    def fit(self, X):
        self.vectorizer.fit(X['statement'])

    def transform(self, X):
        features = []

        # Text features: Bag-of-Words for common words indicative of sensationalism
        statement_features = self.vectorizer.transform(X['statement'])
        features.append(pd.DataFrame(statement_features.toarray(), index=X.index))

        # Writing style and linguistic features

        # 1. Punctuation-based features (Exclamation marks, question marks, ALL CAPS words)
        X['exclamation_count'] = X['statement'].apply(lambda x: x.count('!'))
        X['question_mark_count'] = X['statement'].apply(lambda x: x.count('?'))
        X['all_caps_count'] = X['statement'].apply(lambda x: len(re.findall(r'\b[A-Z]{2,}\b', x)))

        # 2. Named Entity Count: Counts the number of named entities in the statement
        X['entity_count'] = X['statement'].apply(lambda x: len(nlp(x).ents))

        # 3. Superlative and adjective count: Words that amplify sensationalism
        X['superlative_count'] = X['statement'].apply(lambda x: len(re.findall(r'\b(best|worst|most|least)\b', x.lower())))
        X['adjective_count'] = X['statement'].apply(lambda x: len([word for word in TextBlob(x).words if word.pos_tag == 'JJ']))

        # 4. Emotion-based word count
        emotion_words = set(['disaster', 'amazing', 'horrible', 'incredible', 'shocking', 'unbelievable'])
        X['emotion_word_count'] = X['statement'].apply(lambda x: len([word for word in x.lower().split() if word in emotion_words]))

        # 5. Modal verb count
        modal_verbs = set(['might', 'could', 'must', 'should', 'would', 'may'])
        X['modal_verb_count'] = X['statement'].apply(lambda x: len([word for word in x.lower().split() if word in modal_verbs]))

        # 6. Average word length and complex word ratio
        X['avg_word_length'] = X['statement'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        X['complex_word_ratio'] = X['statement'].apply(lambda x: len([word for word in x.split() if len(re.findall(r'[aeiouy]+', word)) > 2]) / (len(x.split()) + 1))

        # 7. Passive voice detection
        X['passive_voice_count'] = X['statement'].apply(lambda x: len(re.findall(r'\bwas\b|\bwere\b|\bbeen\b|\bbeing\b', x.lower())))

        # Sentiment and readability metrics
        X['sentiment_polarity'] = X['statement'].apply(lambda x: TextBlob(x).sentiment.polarity)
        X['sentiment_subjectivity'] = X['statement'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        X['flesch_reading_ease'] = X['statement'].apply(self.flesch_reading_ease)

        # Combine all features
        numerical_features = X[['exclamation_count', 'question_mark_count', 'all_caps_count',
                                'entity_count', 'superlative_count', 'adjective_count',
                                'emotion_word_count', 'modal_verb_count', 'avg_word_length',
                                'complex_word_ratio', 'passive_voice_count',
                                'sentiment_polarity', 'sentiment_subjectivity', 'flesch_reading_ease']]
        features.append(numerical_features)

        return pd.concat(features, axis=1)

    @staticmethod
    def flesch_reading_ease(text):
        # Compute Flesch Reading Ease score for readability analysis
        sentence_count = max(len(re.split(r'[.!?]+', text)), 1)
        word_count = len(text.split())
        syllable_count = sum([SensationalismFeatureGenerator.syllable_count(word) for word in text.split()])
        if word_count > 0:
            return 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        else:
            return 0

    @staticmethod
    def syllable_count(word):
        # Simple syllable count function using regular expressions
        word = word.lower()
        syllables = re.findall(r'[aeiouy]+', word)
        return max(len(syllables), 1)

# Create and fit the feature generator
feature_generator = SensationalismFeatureGenerator()
feature_generator.fit(X_train)

# Apply feature engineering
X_train_features = feature_generator.transform(X_train)
X_test_features = feature_generator.transform(X_test)
