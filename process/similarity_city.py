import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from process.feature_selection import get_feature
from sklearn.metrics.pairwise import cosine_similarity
from skimage import metrics

def get_similarity_df():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    city_dict = {}
    city_poi_percent = {}
    path = '../data'
    selected_columns_name = ['房地产', '自然地物', '行政地标', '旅游景点', '文化传媒', '金融', '运动健身', '道路', '酒店', '休闲娱乐', '政府机构', '丽人', '教育培训', '公司企业', '交通设施']
    print("特征筛选完成， 各特征分类机已就绪")
    scaler = MinMaxScaler()
    for root, dirs, files in os.walk(path):
        for file_name in files:
            city_dict[file_name[:-4]] = pd.read_csv(os.path.join(root, file_name))

    selected_columns_name_plus_location = selected_columns_name + ['纬度', '经度']
    city_type_metrix = {}

    city_y = {}
    #选出相应特征并获得每个city每个特征的分布图 类型为np.array
    for key in city_dict.keys():
        city_y[key] = city_dict[key].loc[:, '急救中心']
        city_dict[key] = city_dict[key].loc[:, selected_columns_name_plus_location]
        city_type_metrix[key] = {}

        #获取各poi type占比
        metrix_x = np.array(city_dict[key].loc[:, selected_columns_name])
        city_poi_percent[key] = (metrix_x.sum(axis=0) / np.sum(metrix_x)).reshape(1, -1)


        #获取各类型的结构图
        for column_name in selected_columns_name:
            whole_number = []
            line_number = []
            for i in range(city_dict[key].shape[0]):
                if i > 0 and city_dict[key].at[i, '纬度'] != city_dict[key].at[i - 1, '纬度']:
                    whole_number.append(line_number)
                    line_number = []
                line_number.append(city_dict[key].at[i, column_name])
            whole_number.append(line_number)
            whole_number = whole_number[::-1]


            #数值缩放到0-255之间方便计算SSIM以获得结构相似度
            final_matrix = np.array(whole_number)
            final_matrix = scaler.fit_transform(final_matrix)
            final_matrix = final_matrix * 255
            final_matrix = final_matrix.astype(np.uint8)

            city_type_metrix[key][column_name] = final_matrix

    similarity_metrix = pd.DataFrame(index=city_dict.keys(), columns=city_dict.keys())
    for key1 in city_dict.keys():
        for key2 in city_dict.keys():
            print("calculating", key1, "between", key2)
            if key1 == key2:
                similarity_metrix.at[key1, key2] = 1

            #计算两个城市之间的相似度
            else:
                # 先计算poi类型占比这样的宏观相似度
                cos_sim = np.squeeze(cosine_similarity(city_poi_percent[key1], city_poi_percent[key2]))
                #再依次计算各个类型的SSIM
                ssim_of_each_col = []
                for column_name in selected_columns_name:
                    #获取矩阵
                    matrix1 = city_type_metrix[key1][column_name]
                    matrix2 = city_type_metrix[key2][column_name]



                    #padding一下, 懒得判断metrix的形状了。写个屎
                    # Calculate the required padding amounts for each dimension
                    pad_rows = (max(matrix1.shape[0], matrix2.shape[0]) - matrix1.shape[0]) / 2
                    pad_columns = (max(matrix1.shape[1], matrix2.shape[1]) - matrix1.shape[1]) / 2

                    # Pad matrix1 to have the same shape as the larger matrix
                    matrix1 = np.pad(matrix1, ((np.ceil(pad_rows).astype(int), np.floor(pad_rows).astype(int)), (np.ceil(pad_columns).astype(int), np.floor(pad_columns).astype(int))), 'constant')

                    # Calculate the required padding amounts for each dimension (for matrix2)
                    pad_rows = (max(matrix1.shape[0], matrix2.shape[0]) - matrix2.shape[0]) / 2
                    pad_columns = (max(matrix1.shape[1], matrix2.shape[1]) - matrix2.shape[1]) / 2

                    # Pad matrix2 to have the same shape as the larger matrix
                    matrix2 = np.pad(matrix2, ((np.ceil(pad_rows).astype(int), np.floor(pad_rows).astype(int)), (np.ceil(pad_columns).astype(int), np.floor(pad_columns).astype(int))), 'constant')


                    #正式计算ssim
                    ssim_of_each_col.append(metrics.structural_similarity(matrix1, matrix2))
                ssim_sim = np.mean(ssim_of_each_col)

                ssim_sim = (ssim_sim + 1) / 2

                similarity_metrix.at[key1, key2] = 0.5 * ssim_sim + 0.5 * cos_sim
    return similarity_metrix, city_dict, city_y

