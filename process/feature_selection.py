from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import fisher_score
import pandas as pd
import numpy as np
import os
from yellowbrick.cluster.elbow import kelbow_visualizer
from sklearn.cluster import KMeans

# beijing = pd.read_csv('data/Beijing.csv')
def I(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    uniq_x, count_x = np.unique(x, return_counts=True)
    uniq_y, count_y = np.unique(y, return_counts=True)
    p_x, p_y = count_x / x.shape[0], count_y / y.shape[0]

    matrix_x, matrix_y = x.reshape(-1, 1), y.reshape(-1, 1)
    collaborative_x_y = np.concatenate((matrix_x, matrix_y), axis=1)
    uniq_col, count_col = np.unique(collaborative_x_y, axis=0, return_counts=True)
    p_xy = count_col / collaborative_x_y.shape[0]

    n = 0
    for scale_x in uniq_x:
        for scale_y in uniq_y:
            px = p_x[np.where(uniq_x == scale_x)[0][0]]
            py = p_y[np.where(uniq_y == scale_y)[0][0]]
            row_index = np.where((uniq_col[:, 0] == scale_x) & (uniq_col[:, 1] == scale_y))[0]
            if row_index.size != 0:
                pxy = p_xy[row_index[0]]
                n += pxy * np.log(pxy / (px * py))
    return n



def MRMR(x, y, range_index, first_feature_index, n_select):
    S = [first_feature_index]

    for i in range(n_select - 1):
        phi = np.random.uniform(-100, -50, size=(x.shape[1],))
        for j in range_index:
            phi[j] = I(x[:, j], y) - (1 / len(S)) * sum([I(x[:, j], x[:, k]) for k in S])
        new_index = np.argmax(phi)
        S.append(new_index)
        range_index.remove(new_index)
        print(i + 1, "/", n_select)


    return S

def fisher_score(x, y):
    y = y.astype(int)
    classes = np.unique(y)
    means = np.array([x[y == cla].mean(axis=0) for cla in classes])
    global_means = x.mean(axis=0)

    S_B = np.zeros(x.shape[1])
    S_W = np.zeros(x.shape[1])

    for cla in classes:
        this_class_x = x[y == cla]
        u_cla = means[cla]

        S_B += (u_cla - global_means) ** 2

        S_W += np.sum((this_class_x - u_cla) ** 2, axis=0) / this_class_x.shape[0]

    return S_B / S_W

# city_py = ['Beijing', 'Changsha', 'Chengdu', 'Guangzhou', 'Hangzhou', 'Hefei', 'Heze', 'Nanjing', 'Ningde', 'Shanghai', 'Shenzhen', 'Tianjin', 'Wuhan', 'Xiamen', 'Xian']
def get_feature():
    city_dict = {}
    path = '../data'
    for root, dirs, files in os.walk(path):
        for file_name in files:
            city_dict[file_name[:-4]] = pd.read_csv(os.path.join(root, file_name))

    whole_cities = np.zeros(shape=(1, 32))
    n = 0
    for key, city_dict[key] in city_dict.items():
        if "水系" in city_dict[key].keys():
            city_dict[key] = city_dict[key].drop(['医疗'], axis=1).reset_index(drop=True)
            a = 6
        else:
            city_dict[key] = city_dict[key].drop(['其他线要素'], axis=1).reset_index(drop=True)

        whole_cities = np.concatenate([whole_cities, city_dict[key]], axis=0)
        n += city_dict[key].shape[0]
    whole_cities = whole_cities[1:, :]

    x = whole_cities[:, :-3]
    y = whole_cities[:, -1]

    y = np.where(y > 1, 1, y)
    # fs = fisher_score.fisher_score(x, y)
    # print(fs)

    fisher = fisher_score(x, y)

    range_index = list(np.where(fisher > 0.1)[0])
    first_select = np.argmax(fisher)


    features_selected = MRMR(x, y, first_feature_index=first_select, range_index=range_index, n_select=15)
    # print(selected_feature)
    # print(np.var(x, axis=0))



    #features_selected,_,_ = MRMR.mrmr(x, y, n_selected_features=15)
    column_name = []


    for i in range(len(features_selected)):
        column_name.append(city_dict['Beijing'].keys()[features_selected[i]])
    print(column_name, features_selected)
    return column_name


# get_feature()





# tag_type = ['美食', '酒店', '购物', '生活服务', '丽人',
#     '旅游景点', '休闲娱乐', '运动健身', '教育培训', '文化传媒',
#     '医疗', '汽车服务', '交通设施', '金融',
#     '房地产', '公司企业', '政府机构', '出入口', '自然地物', '道路', '门址', '绿地', '行政区划', '地铁线路', '公交线路', '行政地标', '商圈', '铁路']
#
# whole_x = np.array(beijing.loc[:, tag_type].astype(int))
# print(whole_x.shape[1])
# whole_y = np.array(beijing.iloc[:, -1].astype(int))
# feature_selected, xxx, xxxx = MRMR.mrmr(whole_x, whole_y, n_selected_features=10)
# real_x = whole_x[:, feature_selected]
# print(real_x.shape[1], xxx, xxxx)
# print(MRMR.mrmr(whole_x, whole_y, n_selected_features=10))
#
# fs = fisher_score.fisher_score(whole_x, whole_y)
# delete_index = []
# for i in range(fs.shape[0]):
#     if fs[i] < 1e-04:
#         delete_index.append(i)
#
# print(delete_index)
# x = np.delete(whole_x, delete_index, axis=1)
# print(x)