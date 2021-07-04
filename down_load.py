import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

dl = nltk.downloader.Downloader('http://nltk.org/nltk_data/index.xml')

# dl = nltk.downloader.Downloader("http://example.com/my_corpus_data/index.xml")
# nltk.download('stopwords')
# dl.download()
# nltk.download()
dl.download('stopwords')


# Got the solution. The issue in my case was that when the NLTK downloader started it had the server index as - http://nltk.github.com/nltk_data/
#
# This needs to be changed to - http://nltk.org/nltk_data/
#
# You can change this by going into the NLTK Downloader window and the File->Change Server Index.