from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english')) | set(["n't", "'s", "nan", "'ve", "ve", "make", "think", "use"])
