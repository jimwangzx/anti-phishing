import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
from urllib.parse import urlparse
from tld import get_tld
import os.path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
import re


# /url_method/malicious_url_by_features.ipynb

#First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

#Length of Top Level Domain
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

#Count of digits
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return -1
    else:
        return 1

#Use of IP or not in domain
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return -1
    else:
        # print 'No matching pattern found'
        return 1

def parse_data():
    urldata = pd.read_csv("../../data/url_features/urldata.csv")

    urldata = urldata.drop('Unnamed: 0', axis=1)


    #Length of URL
    urldata['url_length'] = urldata['url'].apply(lambda i: len(str(i)))

    # Hostname Length
    urldata['hostname_length'] = urldata['url'].apply(lambda i: len(urlparse(i).netloc))

    # Path Length
    urldata['path_length'] = urldata['url'].apply(lambda i: len(urlparse(i).path))

    # First Directory Length
    urldata['fd_length'] = urldata['url'].apply(lambda i: fd_length(i))

    # Length of Top Level Domain
    urldata['tld'] = urldata['url'].apply(lambda i: get_tld(i, fail_silently=True))
    urldata['tld_length'] = urldata['tld'].apply(lambda i: tld_length(i))
    urldata = urldata.drop("tld", 1)

    urldata['count-'] = urldata['url'].apply(lambda i: i.count('-'))
    urldata['count@'] = urldata['url'].apply(lambda i: i.count('@'))
    urldata['count?'] = urldata['url'].apply(lambda i: i.count('?'))
    urldata['count%'] = urldata['url'].apply(lambda i: i.count('%'))
    urldata['count.'] = urldata['url'].apply(lambda i: i.count('.'))
    urldata['count='] = urldata['url'].apply(lambda i: i.count('='))
    urldata['count-http'] = urldata['url'].apply(lambda i: i.count('http'))
    urldata['count-https'] = urldata['url'].apply(lambda i: i.count('https'))
    urldata['count-www'] = urldata['url'].apply(lambda i: i.count('www'))

    urldata['count-digits'] = urldata['url'].apply(lambda i: digit_count(i))

    urldata['count-letters']= urldata['url'].apply(lambda i: letter_count(i))

    urldata['count_dir'] = urldata['url'].apply(lambda i: no_of_dir(i))

    urldata['use_of_ip'] = urldata['url'].apply(lambda i: having_ip_address(i))

    urldata['short_url'] = urldata['url'].apply(lambda i: shortening_service(i))

    print(urldata.shape)

    # Predictor Variables
    x = urldata[['hostname_length',
                 'path_length', 'fd_length', 'tld_length', 'count-', 'count@', 'count?',
                 'count%', 'count.', 'count=', 'count-http', 'count-https', 'count-www', 'count-digits',
                 'count-letters', 'count_dir', 'use_of_ip']]

    # Target Variable
    y = urldata['result']

    # Splitting the data into Training and Testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(x_train, y_train)

    dt_predictions = dt_model.predict(x_test)
    accuracy_score(y_test, dt_predictions)
    print(confusion_matrix(y_test, dt_predictions))
    print(classification_report(y_test, dt_predictions))





if __name__ == '__main__':
    parse_data()