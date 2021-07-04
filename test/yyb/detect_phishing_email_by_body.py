
#!pip install --upgrade coremltools


import coremltools
from coremltools.converters import sklearn
import re, string
import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import LinearSVC as SVC
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
#from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

dict = {'account':42.11758, 'member':19.31017, 'access':17.05149, 'email':16.6749, 'address':15.3276,
             'update':15.3276, 'price':12.55556, 'market':10.23963, 'online':9.770984, 'information':9.474589,
             'work':8.331161, 'credit':5.883471, 'response':4.504742, 'offer':3.3573, 'transaction':2.809165,
             'agreement':1.863823, 'registration':1.791679, 'person':1.522086, 'system':1.276965,
             'process':1.076627, 'service':0.895862, 'request':0.616316, 'message':0.32805}

key_words = ['account', 'member', 'access', 'email', 'address',
             'update', 'price', 'market', 'online', 'information',
             'work', 'credit', 'response', 'offer', 'transaction',
             'agreement', 'registration', 'person', 'system',
             'process', 'service', 'request', 'message']

no_head_train_path_0 = '../../data/phishing/IWSPA-AP-traindata/phish/'
no_head_train_path_1 = '../../data/phishing/IWSPA-AP-traindata/legit/'
no_head_test_path = '../../data/phishing/IWSPA-APTestData/testdata_noheaders/'


class MailContentsFeaturize(object):
    def __init__(self, content):
        self.content = content.lower()

    # 提取特征方法
    def contain_key(self, key):
        if key in self.content:
            # return int(1)
            return float(dict[key])
        else:
            return float(0)

    def get_train_data(self):
        data = []
        for word in key_words:
            data.append(self.contain_key(word))
        return data


def clean_text(text):
    text = text.encode('utf-8').decode('utf-8')
    while '\n' in text:
        text = text.replace('\n', ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')
    words = text.split()
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    stripped = []
    for token in words:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            stripped.append(new_token.lower())
    text = ' '.join(stripped)
    return text


def get_data(path):
    text_list = list()
    files = os.listdir(path)
    for text_file in files:
        file_path = os.path.join(path, text_file)
        read_file = open(file_path, 'r+')
        read_text = read_file.read()
        read_file.close()
        cleaned_text = clean_text(read_text)
        # append text
        # text_list.append(cleaned_text)

        featurized = MailContentsFeaturize(cleaned_text)
        # append features
        # text_list.append(featurized.run())
        text_list.append(featurized.get_train_data())
    return text_list, files


def generate_test_data():
    no_head_train_0, temp = get_data(no_head_train_path_0)
    no_head_train_1, temp = get_data(no_head_train_path_1)

    no_head_train = no_head_train_0 + no_head_train_1
    no_head_labels_train = ([int(0)] * len(no_head_train_0)) + ([int(1)] * len(no_head_train_1))

    shuffled_indices = np.random.permutation(len(no_head_labels_train))
    train_data = np.array(no_head_train)[shuffled_indices]
    train_data = train_data.tolist()
    train_label = np.array(no_head_labels_train)[shuffled_indices]
    train_label = train_label.tolist()

    temp_train_data = train_data[0:int(0.8 * len(train_data))]
    temp_train_label = train_label[0:int(0.8 * len(train_label))]
    temp_test_data = train_data[int(0.8 * len(train_data)):]
    temp_test_labels = train_label[int(0.8 * len(train_label)):]
    return temp_train_data, temp_train_label, temp_test_data, temp_test_labels

########################################################################################################################
def generate_dt_model():
    temp_train_data, temp_train_label, temp_test_data, temp_test_labels = generate_test_data()

    dt_model = DecisionTreeClassifier()
    dt_model.fit(temp_train_data, temp_train_label)

    coreml_model = sklearn.convert(dt_model, key_words, 'output')
    coreml_model.save('coreml_dt.mlmodel')

    dt_predictions = dt_model.predict(temp_test_data)
    accuracy_score(temp_test_labels, dt_predictions)

    print(confusion_matrix(temp_test_labels, dt_predictions))
    print(classification_report(temp_test_labels, dt_predictions))

def verify_dt_model():
    model = coremltools.models.MLModel('coreml_dt.mlmodel')
    # model.predict({'input': temp_test_data})
    pred_out_put = model.predict(
        {'account': 42.11758, 'member': 19.31017, 'access': 17.05149, 'email': 16.6749, 'address': 15.3276,
         'update': 15.3276, 'price': 12.55556, 'market': 10.23963, 'online': 9.770984, 'information': 9.474589,
         'work': 8.331161, 'credit': 5.883471, 'response': 4.504742, 'offer': 3.3573, 'transaction': 2.809165,
         'agreement': 1.863823, 'registration': 1.791679, 'person': 1.522086, 'system': 1.276965,
         'process': 1.076627, 'service': 0.895862, 'request': 0.616316, 'message': 0.32805})
    print(pred_out_put)
    print(pred_out_put["output"])

########################################################################################################################
# Logistic Regression
def logistic_regression():
    temp_train_data, temp_train_label, temp_test_data, temp_test_labels = generate_test_data()
    log_model = LogisticRegression()
    log_model.fit(temp_train_data, temp_train_label)
    log_predictions = log_model.predict(temp_test_data)
    accuracy_score(temp_test_labels, log_predictions)
    print("Logistic Regression")
    print(confusion_matrix(temp_test_labels, log_predictions))
    print(classification_report(temp_test_labels, log_predictions))

########################################################################################################################

###create decision tree classifier object
def generate_dtree_model():
    temp_train_data, temp_train_label, temp_test_data, temp_test_labels = generate_test_data()
    DT = tree.DecisionTreeClassifier(criterion="gini", max_depth=4)
    ##fit decision tree model with training data
    DT.fit(temp_train_data, temp_train_label)
    ##test data prediction
    DT_expost_preds = DT.predict(temp_test_data)
    accuracy_score(temp_test_labels, DT_expost_preds)

    print("gini decision tree classifier ")

    print(confusion_matrix(temp_test_labels, DT_expost_preds))
    print(classification_report(temp_test_labels, DT_expost_preds))

    dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
    dtree.fit(temp_train_data, temp_train_label)

    coreml_model = sklearn.convert(dtree, key_words, 'output')
    coreml_model.save('coreml_dtree.mlmodel')

    pred = dtree.predict(temp_test_data)
    accuracy_score(temp_test_labels, pred)

    print("entropy decision tree classifier ")
    print(confusion_matrix(temp_test_labels, pred))
    print(classification_report(temp_test_labels, pred))

def verify_dtree_model():
    model = coremltools.models.MLModel('coreml_dtree.mlmodel')
    # model.predict({'input': temp_test_data})
    pred_out_put = model.predict(
        {'account': 42.11758, 'member': 19.31017, 'access': 17.05149, 'email': 16.6749, 'address': 15.3276,
         'update': 15.3276, 'price': 12.55556, 'market': 10.23963, 'online': 9.770984, 'information': 9.474589,
         'work': 8.331161, 'credit': 5.883471, 'response': 4.504742, 'offer': 3.3573, 'transaction': 2.809165,
         'agreement': 1.863823, 'registration': 1.791679, 'person': 1.522086, 'system': 1.276965,
         'process': 1.076627, 'service': 0.895862, 'request': 0.616316, 'message': 0.32805})
    print(pred_out_put["output"])
    print(pred_out_put)

########################################################################################################################
def generate_svm_model():
    temp_train_data, temp_train_label, temp_test_data, temp_test_labels = generate_test_data()
    svc = SVC()
    inputs = np.asarray(temp_train_data, dtype=float)
    outputs = np.asarray(temp_train_label, dtype=int)
    model_5 = svc.fit(inputs, outputs)

    coreml_model = sklearn.convert(model_5, key_words, 'output')
    coreml_model.save('coreml_svm.mlmodel')

    pred = model_5.predict(temp_test_data)
    accuracy_score(temp_test_labels, pred)

    print("svm ")
    print(confusion_matrix(temp_test_labels, pred))
    print(classification_report(temp_test_labels, pred))

def verify_svm_model():
    model = coremltools.models.MLModel('coreml_svm.mlmodel')
    pred_out_put = model.predict(
        {'account': 42.11758, 'member': 19.31017, 'access': 17.05149, 'email': 16.6749, 'address': 15.3276,
         'update': 15.3276, 'price': 12.55556, 'market': 10.23963, 'online': 9.770984, 'information': 9.474589,
         'work': 8.331161, 'credit': 5.883471, 'response': 4.504742, 'offer': 3.3573, 'transaction': 2.809165,
         'agreement': 1.863823, 'registration': 1.791679, 'person': 1.522086, 'system': 1.276965,
         'process': 1.076627, 'service': 0.895862, 'request': 0.616316, 'message': 0.32805})
    print(pred_out_put["output"])
    print(pred_out_put)

########################################################################################################################
def generate_boost_model():
    print("AdaBoostClassifier ")
    temp_train_data, temp_train_label, temp_test_data, temp_test_labels = generate_test_data()
    adc = AdaBoostClassifier(n_estimators=5, learning_rate=1)
    model_6 = adc.fit(temp_train_data, temp_train_label)

    pred = model_6.predict(temp_test_data)
    accuracy_score(temp_test_labels, pred)
    print(confusion_matrix(temp_test_labels, pred))
    print(classification_report(temp_test_labels, pred))


########################################################################################################################
def generate_xgboost_model():
    print("xgboost ")
    temp_train_data, temp_train_label, temp_test_data, temp_test_labels = generate_test_data()
    xgb = XGBClassifier()
    # model_7=xgb.fit(temp_train_data, temp_train_label)
    # pred = model_7.predict(temp_test_data)
    model_7 = xgb.fit(np.asarray(temp_train_data, dtype=float), np.asarray(temp_train_label, dtype=int))
    pred = model_7.predict(np.asarray(temp_test_data))

    accuracy_score(temp_test_labels, pred)
    print(confusion_matrix(temp_test_labels, pred))
    print(classification_report(temp_test_labels, pred))

########################################################################################################################
def generate_ababoost_model():
    print("AdaBoostClassifier ")
    temp_train_data, temp_train_label, temp_test_data, temp_test_labels = generate_test_data()
    gb = GradientBoostingClassifier()
    model_gb = gb.fit(temp_train_data, temp_train_label)

    # coreml_model = sklearn.convert(model_gb, key_words, 'output')
    # coreml_model.save('coreml_gb.mlmodel')

    pred = model_gb.predict(temp_test_data)
    accuracy_score(temp_test_labels, pred)

    print(confusion_matrix(temp_test_labels, pred))
    print(classification_report(temp_test_labels, pred))

if __name__ == '__main__':
    generate_dt_model()
    #verify_dt_model()
    #logistic_regression()
    generate_dtree_model()
    #verify_dtree_model()
    generate_svm_model()
    #verify_svm_model()
    #generate_boost_model()
    #generate_xgboost_model()
    #generate_ababoost_model()