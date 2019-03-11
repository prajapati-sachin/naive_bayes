import nltk
from nltk.tokenize import word_tokenize

text = "to be or not to be"
tokens = nltk.word_tokenize(text)
# print(type(tokens))
# print(type(text.split()))
bigrm = nltk.bigrams(tokens)
# print(list(bigrm))
print(*map(' '.join, bigrm), sep=', ')
