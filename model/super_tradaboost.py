import numpy as np
from tradaboost import Tradaboost
from process.similarity_city import get_similarity_df


class super_tradaboost(object):
    def __init__(self, M):
        self.M = M
        print(self.M)
        self.estimators = []
        self.weights_of_estimator = []
        self.all_weight = []
        self.bata = None

    def calculate_weights(self, weights):
        whole_weights = np.sum(weights)
        return np.asarray(whole_weights / weights, order="C")

    def predict(self, x):
        pred_y = np.zeros(shape=(x.shape[0]))
        for i in range(self.M):
            this_time_weight = self.all_weight[:, i].reshape(-1, 1)
            total = np.sum(this_time_weight)
            this_time_weight /= total
            this_time_bata = self.bata[0, i]
            allresult = []
            for j in range(len(self.estimators)):
                h_vector = self.estimators[j].predict(x)
                allresult.append(h_vector)

            allresult = np.array(allresult).T
            pred_y += (allresult @ this_time_weight).reshape(-1) * this_time_bata
        pred_y = np.where(pred_y > 0, 1, pred_y)
        return pred_y

    def estimators_predict(self, x):
        estimator_weight_np = np.array(self.weights_of_estimator)
        allresult = []
        for j in range(len(self.estimators)):
            h_vector = self.estimators[j].predict(x)
            allresult.append(h_vector)

        allresult = np.array(allresult).T

        return (allresult @ estimator_weight_np).reshape(-1)


    def change_estimators_weight(self, target_x, target_y, weight_target):
        for i in range(len(self.estimators)):
            each_prediction = self.estimators[i].predict(target_x)
            each_error = self.calculate_error_rate(target_y, each_prediction, weight_target)
            each_error = 0.5 if each_error >= 0.5 else each_error
            self.weights_of_estimator[i] = np.log((1 - each_error) / each_error) / 2


    def phase_one(self, multi_x, multi_y, target_x, target_y):

        for i in range(len(multi_x)):
            tradaboost = Tradaboost(N=self.M)
            this_estimator, this_weight = tradaboost.fit(multi_x[i], target_x, multi_y[i], target_y)
            self.estimators.append(this_estimator)
            self.weights_of_estimator.append(this_weight)

        self.weights_of_estimator = np.array(self.weights_of_estimator)
        self.weights_of_estimator = self.weights_of_estimator.reshape(-1, 1)
        self.all_weight = np.zeros([self.weights_of_estimator.shape[0], self.M])



    def calculate_error_rate(self, label, predict, weights):
        whole_weights = np.sum(weights)
        return np.sum(weights[:, 0] * np.abs(label - predict) / whole_weights)


    def phase_two(self, target_x, target_y):


        target_line_number = target_x.shape[0]
        weight_target = np.ones([target_line_number, 1]) / target_line_number

        bata_T = np.zeros([1, self.M])


        for i in range(self.M):
            # P = self.calculate_weights(weight_target)

            self.change_estimators_weight(target_x, target_y, weight_target)
            self.all_weight[:, i] = self.weights_of_estimator.copy().reshape(-1)

            res = self.estimators_predict(target_x)
            error = self.calculate_error_rate(target_y, res, weight_target)

            error = 0.5 if error >= 0.5 else error
            if error == 0:
                print("咋可能")
            print(error)
            bata_T[0, i] = np.log((1 - error) / error) / 2

            weight_target[:, 0] *= np.exp(-bata_T[0, i] * res * target_y)



        self.bata = bata_T
        total = np.sum(self.bata)
        self.bata /= total

    def fit(self, multi_x, multi_y, target_x, target_y):
        self.phase_one(multi_x, multi_y, target_x, target_y)
        self.phase_two(target_x, target_y)
        print(self.all_weight)
        print(self.bata)





def get_test_multi(n):
    multi_x = []
    multi_y = []
    for i in range(n):
        this = np.random.randint(1500, 2500)
        source_x = np.random.uniform(0, 600, size=(this, 20))

        source_label = np.zeros((this,), dtype=int)
        set_one = np.random.choice(this, size=int(this * 0.1), replace=False)
        source_label[set_one] = 1
        multi_x.append(source_x)
        multi_y.append(source_label)
    return multi_x, multi_y

similarity_matrix, city_dict, city_y = get_similarity_df()
heze = similarity_matrix.loc[:, 'Heze']
heze.drop(['Heze', 'Ningde'], axis=0, inplace=True)


selected_city_that_has_high_similarity_with_heze = heze[heze > 0.77].index.tolist()

multix, multiy = [], []
city_location = {}

for each_city in city_dict.keys():
    city_location[each_city] = city_dict[each_city].loc[:, ['纬度', '经度']]
    city_dict[each_city] = city_dict[each_city].drop(['纬度', '经度'], axis=1).reset_index(drop=True)
    city_y[each_city] = np.array(city_y[each_city])
    city_y[each_city] = np.where(city_y[each_city] > 1, 1, city_y[each_city])

for each_city in selected_city_that_has_high_similarity_with_heze:
    multix.append(np.array(city_dict[each_city]))
    multiy.append(city_y[each_city])

for i in range(len(multix)):
    print(multix[i].shape, multiy[i].shape)

heze_x = np.array(city_dict['Heze'])
heze_y = city_y['Heze']

heze_local = np.array(city_location["Heze"])

supertradaboost = super_tradaboost(20)

supertradaboost.fit(multix, multiy, heze_x, heze_y)
pred_y = supertradaboost.predict(heze_x)
print(np.unique(pred_y))
print(pred_y.shape, pred_y[pred_y > 0], heze_local[pred_y > 0])

# multix, multiy = get_test_multi(3)
#
# test = super_tradaboost(10)
# target_x = np.random.uniform(0, 300, size=(300, 20))
#
# target_label = np.zeros((300,), dtype=int)
# target_set_one = np.random.choice(200, size=int(300 * 0.1), replace=False)
# target_label[target_set_one] = 1
# test.fit(multix, multiy, target_x, target_label)