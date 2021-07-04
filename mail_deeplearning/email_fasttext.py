import os, re, string
import numpy as np
import fasttext
import time


def train_model(ipt=None, opt=None, model='', dim=100, epoch=5, lr=0.1, loss='softmax'):
	np.set_printoptions(suppress=True)
	if os.path.isfile(model):
		classifier = fasttext.load_model(model)
	else:
		print('run here')
		classifier = fasttext.train_supervised(ipt, label='__label__', dim=dim, epoch=epoch,
		                                       lr=lr, wordNgrams=2, loss=loss)
		classifier.save_model(opt)
	return classifier


class _MD(object):
	mapper = {
		str: '',
		int: 0,
		list: list,
		dict: dict,
		set: set,
		bool: False,
		float: .0
	}

	def __init__(self, obj, default=None):
		self.dict = {}
		assert obj in self.mapper, \
			'got a error type'
		self.t = obj
		if default is None:
			return
		assert isinstance(default, obj), \
			f'default ({default}) must be {obj}'
		self.v = default

	def __setitem__(self, key, value):
		self.dict[key] = value

	def __getitem__(self, item):
		if item not in self.dict and hasattr(self, 'v'):
			self.dict[item] = self.v
			return self.v
		elif item not in self.dict:
			if callable(self.mapper[self.t]):
				self.dict[item] = self.mapper[self.t]()
			else:
				self.dict[item] = self.mapper[self.t]
			return self.dict[item]
		return self.dict[item]


def defaultdict(obj, default=None):
	return _MD(obj, default)


def cal_precision_and_recall(file='data_test.txt'):
	precision = defaultdict(int, 1)
	recall = defaultdict(int, 1)
	total = defaultdict(int, 1)
	with open(file) as f:
		for line in f:
			if(line == "\n"):
				break
			label, content = line.split(' ', 1)
			# print(label)
			# print(content)
			total[label.strip().strip('__label__')] += 1
			labels2 = classifier.predict(content.replace('\n', ''))
			# print(labels2)
			pre_label, sim = labels2[0][0], labels2[1][0]
			recall[pre_label.strip().strip('__label__')] += 1

			if label.strip() == pre_label.strip():
				precision[label.strip().strip('__label__')] += 1

	print('precision', precision.dict)
	print('recall', recall.dict)
	print('total', total.dict)

	for sub in precision.dict:
		print(total[sub])
		pre = precision[sub] / total[sub]
		rec = precision[sub] / recall[sub]
		F1 = (2 * pre * rec) / (pre + rec)
		print(f"{sub.strip('__label__')}  precision: {str(pre)}  recall: {str(rec)}  F1: {str(F1)}")


if __name__ == '__main__':
	start_time = time.time()

	fast_train_file = './data/fast_train.txt'
	fast_test_file = './data/fast_test.txt'
	fast_test_file_wc_normal = "./data/edison_normal.txt" #"./data/combine_txt_wc_normal.txt"
	fast_test_file_wc_phishing = "./data/edison_phishing.txt"
	fast_test_file_jose_phishing = "./data/jose_phishing.txt"
	fast_test_file_merger_wc = "./data/edison.txt"

	dim=10
	epoch=10
	lr=0.1
	loss='softmax'
	#修改为自己的地址，主要是为了不显著增加项目的大小
	# model_path =  f'/home/anbo/email/models/fasttext/data_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.model'
	# model_file_path = f'~/email/models/fasttext/data_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.model'

	import os
	home_path = os.environ['HOME']
	saved_path = home_path + '/email/models/fasttext/'

	if not os.path.exists(saved_path):
		os.makedirs(saved_path)
	model = saved_path + f'data_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.model'
	classifier = train_model(ipt=fast_train_file,
						 opt=model,
						 model=model,
						 dim=dim, epoch=epoch, lr=0.5
						 )
	print('==================fast_test_file=============================')

	result = classifier.test(fast_test_file)
	print(result)
	#(1145, 0.9877729257641922, 0.9877729257641922) (测试数据量，precision，recall)
	cal_precision_and_recall(fast_test_file)
	'''
	1015
	1  precision: 0.9960591133004926  recall: 0.990205680705191  F1: 0.9931237721021611
	132
	0  precision: 0.9242424242424242  recall: 0.9682539682539683  F1: 0.9457364341085271
	'''

	print('time is :' + str(time.time()- start_time))
	#time is :0.23511672019958496

	print('\n======================fast_test_file_merger_wc=============================')

	# =========================================================

	result_merge_wc = classifier.test(fast_test_file_merger_wc)
	print(result_merge_wc)
	'''
	(36, 0.5555555555555556, 0.5555555555555556)
	precision {'1': 17, '0': 5}
	recall {'1': 29, '0': 9}
	total {'1': 21, '0': 17}
	21
	1  precision: 0.8095238095238095  recall: 0.5862068965517241  F1: 0.68
	17
	0  precision: 0.29411764705882354  recall: 0.5555555555555556  F1: 0.3846153846153846
	'''
	cal_precision_and_recall(fast_test_file_merger_wc)

	print('\n=======================fast_test_file_wc_normal==================================')

	# =========================================================

	result_wc_normal = classifier.test(fast_test_file_wc_normal)
	print(result_wc_normal)
	# (1145, 0.9877729257641922, 0.9877729257641922) (测试数据量，precision，recall)
	cal_precision_and_recall(fast_test_file_wc_normal)
	'''
	(20, 0.8, 0.8)
	precision {'1': 17}
	recall {'1': 17, '0': 5}
	total {'1': 21}
	21
	1  precision: 0.8095238095238095  recall: 1.0  F1: 0.8947368421052632
    '''

	print('\n==========================fast_test_file_wc_phishing===============================')

	# =============================================================

	result_wc_phishing = classifier.test(fast_test_file_wc_phishing)
	print(result_wc_phishing)
	# (1145, 0.9877729257641922, 0.9877729257641922) (测试数据量，precision，recall)
	cal_precision_and_recall(fast_test_file_wc_phishing)
	'''
	(16, 0.25, 0.25)
	precision {'0': 5}
	recall {'1': 13, '0': 5}
	total {'0': 17}
	17
	0  precision: 0.29411764705882354  recall: 1.0  F1: 0.45454545454545453
    '''

	print('\n====================fast_test_file_jose_phishing=====================================')

	#
	result_jose_phishing = classifier.test(fast_test_file_jose_phishing)
	print(result_jose_phishing)
	# (1145, 0.9877729257641922, 0.9877729257641922) (测试数据量，precision，recall)
	cal_precision_and_recall(fast_test_file_jose_phishing)
	'''
	precision {'0': 8112}
	recall {'1': 2037, '0': 8112}
	total {'0': 10148}
	10148
	0  precision: 0.7993693338588884  recall: 1.0  F1: 0.8884994523548739
	'''



