# ant-phishing
## 计划
|  序号   | 问题  | 文件 | 状态 | 是否可以mobile | 是否最佳 |
|  ----  | ----  | ----  | ----  | ----  | ----  |
| 1  | url cnn分类 | malicious_cnn.ipynb | 完成 | 可以  | ----  |
| 2  | url attention | malicious_attention.ipynd  | 完成 | ----  | ----  |
| 3  | url features decision tree | malicious_url_by_features.ipynb  | 完成 | 有难度  | ----  |
| 4  | url  decision tree | decision_tree_for_urls.ipynb  | 完成 | ----  | 非常好  |
| 5  | html structure0 naive bayes | phishing_naive_bayes.ipynd  | 完成 | ----  | ----  |
| 6  | html structure0 decision tree | phishing_decision_tree.ipynd  | 完成 | ----  | ----  |
| 7  | html structure1  ensemble | structure1_ensemble.ipynb  | 完成 | ----  | ----  |
| 8  | fraudulent bert | by_bert.ipynb  | 完成 | ----  | ----  |
| 9  | fraudulent elmo | by_elmo.ipynb  | 完成 | ----  | ----  |
| 10  | fraudulent for email | email.ipynb  | 完成 | ----  | 利用词特征效果很好  |
| 11  | fraudulent multi_task | Multi-task_Learning.ipynb  | 完成 | ----  | ----  |
| 12  | phishing dataset classical and fasttext | IWSPA.ipynd  | 完成 | header + noheader  | ----  |
| 13  | text words features | text_feature_1.ipynb  | 完成 | noheader  | ----  |
| 14  | robust nlp  | robust_nlp.ipynd  | 进行中 | ----  | ----  |
| 15  | html + header  | malicious_attention.ipynd  | 未完成 | ----  | ----  |
| 16  | semantic | malicious_attention.ipynd  | 未完成 | ----  | ----  |
| 17  | visual similarity | malicious_attention.ipynd  | 未完成 | ----  | ----  |
#### 准确率直接去文件看。
### 思考
* bert适合解决这个问题吗？
* intent分类
    > 一般越迫切越被怀疑
* 多维数据融合
* 个人语法特征
* fraudulent和mail_method是一套py环境，其他是另一套。