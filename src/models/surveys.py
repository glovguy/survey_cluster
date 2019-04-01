import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from constants.stopwords import STOPWORDS


class Surveys(object):
    stemmer = nltk.stem.snowball.SnowballStemmer("english")

    def __init__(self, data_raw):
        self.data_raw = data_raw
        self.survey_text = [r['survey_text'] for r in data_raw]
        self.vectorizer = TfidfVectorizer(**self.vectorizer_options())
        self.tfidf_matrix = self.vectorizer.fit_transform(self.survey_text)
        self.set_tfidf_feature_to_word()

    def vectorizer_options(self):
        return {
                'max_df': 0.8,
                'min_df': 0.03,
                'max_features': 200000,
                'binary': True,
                'stop_words': 'english',
                'use_idf': True,
                'tokenizer': self.tokenize_and_stem,
                'ngram_range': (1,3)
            }

    def set_tfidf_feature_to_word(self):
        vocab = [token for review in self.survey_text for token in self.tokenize(review)]
        vocab_stemmed = [token for review in self.survey_text for token in self.tokenize_and_stem(review)]
        self.tfidf_feature_to_word = {vocab_stemmed[i]: vocab[i] for i in range(0, len(vocab_stemmed))}

    def survey_count(self):
        return len(self.survey_text)

    def feature_names_unstemmed(self):
        output = []
        for feature in self.vectorizer.get_feature_names():
            output.append(' '.join([self.tfidf_feature_to_word[w] for w in feature.split(' ')]))
        return output

    @classmethod
    def is_not_number_or_stopword(cls, token):
        return re.search('[a-zA-Z]', token) and token not in STOPWORDS

    @classmethod
    def tokenize_and_stem(cls, text):
        filtered_tokens = cls.tokenize(text)
        stems = [cls.stemmer.stem(t) for t in filtered_tokens]
        return stems

    @classmethod
    def tokenize(cls, text):
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        return cls.filter_tokens(tokens)

    @classmethod
    def filter_tokens(cls, tokens):
        return [str(token) for token in tokens if cls.is_not_number_or_stopword(token)]
