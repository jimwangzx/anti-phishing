import os, re, string
import numpy as np

import time



edision_norm_data = '../test_data/from_weicheng/normal_txt/'
edision_phishing_data = '../test_data/from_weicheng/phishing_txt/'
jose_phishing_data = "../test_data/jose-phishing/"


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
		if not text_file.endswith(".txt"):
			continue
		# print(text_file)
		file_path = os.path.join(path, text_file)
		# print(file_path)
		read_file = open(file_path, 'r+')
		read_text = read_file.read()
		read_file.close()
		cleaned_text = clean_text(read_text)
		text_list.append(cleaned_text)
	return text_list, files


no_head_train_0, temp = get_data(edision_phishing_data)
no_head_train_1, temp = get_data(edision_norm_data)
no_head_train_jose, temp = get_data(jose_phishing_data)


fast_train_file = './data/edison_phishing.txt'
fast_test_file = './data/edison_normal.txt'

mergefile =  './data/edison.txt'
jose_phishing_file = "./data/jose_phishing.txt"

writeFile = open(fast_train_file, 'w')
for text, label in zip(no_head_train_0, [0]*len(no_head_train_0)):
	writeFile.write('__label__'+str(label)+' '+str(text.encode('utf-8'))+'\n')
writeFile.close()


writeFile = open(fast_test_file, 'w')
for text, label in zip(no_head_train_1, [1]*len(no_head_train_1)):
	writeFile.write('__label__'+str(label)+' '+str(text.encode('utf-8'))+'\n')
writeFile.close()



writeFile = open(mergefile, 'w')
for text, label in zip(no_head_train_1, [1]*len(no_head_train_1)):
	writeFile.write('__label__'+str(label)+' '+str(text.encode('utf-8'))+'\n')
for text, label in zip(no_head_train_0, [0]*len(no_head_train_0)):
	writeFile.write('__label__'+str(label)+' '+str(text.encode('utf-8'))+'\n')
writeFile.close()


writeFile = open(jose_phishing_file, 'w')
for text, label in zip(no_head_train_jose, [0]*len(no_head_train_jose)):
	writeFile.write('__label__'+str(label)+' '+str(text.encode('utf-8'))+'\n')
writeFile.close()




if __name__ == '__main__':
	pass