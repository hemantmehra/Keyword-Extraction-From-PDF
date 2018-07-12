import numpy as np
# For dataframe
import pandas as pd

# For English Stopwords
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# For Tf-idf Weighting
from sklearn.feature_extraction.text import TfidfVectorizer

# For Regular Expressions
import re


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

from io import StringIO

# parsing pdf file

def pdf_to_text(pdfname):

    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Extract text
    fp = open(pdfname, 'rb')
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
    fp.close()

    # Get text from StringIO
    text = sio.getvalue()

    # Cleanup
    device.close()
    sio.close()

    return text

extracted_text = pdf_to_text("JavaBasics-notes.pdf")

# Preprocessing Text
def preprocess_text(text):
    tokens = word_tokenize(text)
    words = []
    for word in tokens:
        # Replace non-alpha character with space
        word = re.sub('[^A-Za-z]+', ' ', word)
        words.append(word.lower())
    return " ".join(words)

stop_words = stopwords.words('english')

stop_words.extend(["java", "jguru", "com", "all", "rights",
                   "reserved", "etc", "abc", "hello",
                   "world", "www", "blah"])


# Using sklearn Tfidf Vectorizer
vectorizer = TfidfVectorizer(stop_words = stop_words, 
                             preprocessor=preprocess_text)
X = vectorizer.fit_transform([extracted_text])
words = vectorizer.get_feature_names()
weights = X.toarray()


ordered_weights = weights[0][(np.argsort(-weights[0]))]
weight_ordered_words = np.array(words)[np.argsort(-weights[0])]

# Creating pandas Dataframe

d = {"words": weight_ordered_words, "weight": ordered_weights}
df = pd.DataFrame(data = d, columns = ["words", "weight"])

# Outputing to csv file
df.to_csv("keywords.csv", encoding='utf-8', index=False)