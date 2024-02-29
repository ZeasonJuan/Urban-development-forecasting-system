import numpy as np
from sklearn.metrics import roc_auc_score
from model.mlp import MLP
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Tradaboost(object):
    def __init__(self, N = None, threshold = None,score = roc_auc_score):
        self.N = N
        self.threshold = threshold
        self.score = score
        self.estimators = []

    def calculate_weights(self, weights):
        whole_weights = np.sum(weights)
        return np.asarray(whole_weights / weights, order="C")

    def calculate_error_rate(self, label, predict, weights):
        whole_weights = np.sum(weights)
        return np.sum(weights[:, 0] * np.abs(label - predict) / whole_weights)

    def fit(self, source, target, source_label, target_label):
        source_lines_number = source.shape[0]
        target_lines_number = target.shape[0]

        all_data = np.concatenate((source, target), axis=0)
        all_label = np.concatenate((source_label, target_label), axis=0)

        #初始化, maybe different
        weight_target = np.ones([target_lines_number, 1]) / target_lines_number
        weight_source = np.ones([source_lines_number, 1]) / source_lines_number

        weights = np.concatenate((weight_source, weight_target), axis=0)


        bata = 1 / (1 + np.sqrt(2 * np.log(source_lines_number / self.N)))
        bata_T = np.zeros([1, self.N])

        all_results = np.ones([source_lines_number + target_lines_number, self.N])

        all_data = np.asarray(all_data, order="C")
        all_label = np.asarray(all_label, order="C")
        for i in range(self.N):
            #标准化weights
            P = self.calculate_weights(weights)

            #训练基分类模型
            mlp = MLP()
            mlp.fit(all_data * P, all_label)
            self.estimators.append(mlp)

            #储存对所有的预测结果
            predict_on_all_data = mlp.predict(all_data)
            all_results[:, i] = predict_on_all_data

            #计算在目标域的错误率
            predict_on_target = mlp.predict(target)

            error_rate = self.calculate_error_rate(target_label, predict_on_target, weights[source_lines_number: target_lines_number + source_lines_number, :])

            #得到bata值, 由定义得到的
            error_rate = 0.49 if error_rate >= 0.5 else error_rate


            if error_rate == 0:
                print('All right?')
                break
            bata_T[0, i] = np.log((1 - error_rate) / error_rate) / 2

            #更新weights_source
            weights[0: source_lines_number, 0] *= np.power(bata, np.abs(all_results[0: source_lines_number, i] - source_label))

            #更新weights_target
            weights[source_lines_number: target_lines_number + source_lines_number, 0] *= np.power(bata_T[0, i], -np.abs(all_results[source_lines_number: target_lines_number + source_lines_number, i] - target_label))

            score_this_round = self.score(target_label, predict_on_target)
            print(i, '轮的分数为', bata_T[0, i])
        return self.estimators[np.argmax(bata_T)], np.max(bata_T)


# source_x = np.random.uniform(0, 600, size=(2000, 20))
#
# source_label = np.zeros((2000,), dtype=int)
# set_one = np.random.choice(2000, size=int(2000 * 0.1), replace=False)
# source_label[set_one] = 1
#
# target_x = np.random.uniform(0, 300, size=(300, 20))
#
# target_label = np.zeros((300,), dtype=int)
# target_set_one = np.random.choice(200, size=int(300 * 0.1), replace=False)
# target_label[target_set_one] = 1
#
#
# tradaboost = Tradaboost(N=50)
# tradaboost.fit(source_x, target_x, source_label=source_label, target_label=target_label)





