import csv

import pymysql
import matplotlib.pyplot as plt
from geopy.distance import geodesic as distence
import numpy as np
import torch
import time
import math
import cupy as cp




def which_one_minous_most(decrease_list, is_decrease):
    if is_decrease:
        grad_list = [decrease_list[x] - decrease_list[x + 1] for x in range(len(decrease_list) - 1)]
        return grad_list.index(max(grad_list))

def new_get_border(lat_list, lng_list):
    max_lat = max(lat_list)
    min_lat = min(lat_list)
    max_lng = max(lng_list)
    min_lng = min(lng_list)
    south_to_north_distance = distence((min_lat, max_lng), (max_lat, max_lng))
    west_to_east_distance = distence((min_lat, min_lng), (min_lat, max_lng))
    print(south_to_north_distance, west_to_east_distance, max_lng, min_lng, min_lat, max_lat)
    south_to_north_distance = float(south_to_north_distance.km)
    west_to_east_distance = float(west_to_east_distance.km)

    one_km_lat = (max_lat - min_lat) / south_to_north_distance
    one_km_lng = (max_lng - min_lng) / west_to_east_distance

    #找到最高点
    poi_matrix = np.zeros(shape=(round(south_to_north_distance), round(west_to_east_distance)))
    for i in range(round(south_to_north_distance)):
        down_lat_this_section = min_lat + i * one_km_lat
        up_lat_this_section = min_lat + (i + 1) * one_km_lat
        for j in range(round(west_to_east_distance)):
            left_lng_this_section = min_lng + j * one_km_lng
            right_lng_this_section = min_lng + (j + 1) * one_km_lng
            n = 0
            print(i * poi_matrix.shape[1] + j, "/", poi_matrix.shape[0] * poi_matrix.shape[1])
            for k in range(len(lat_list)):
                if down_lat_this_section <= lat_list[k] < up_lat_this_section:
                    if left_lng_this_section <= lng_list[k] < right_lng_this_section:
                        n += 1

            poi_matrix[i, j] = n

    max_row, max_col = np.unravel_index(np.argmax(poi_matrix), poi_matrix.shape)
    print("here")
    local_list = ramble_border(poi_matrix, max_row, max_col, 0.70)
    print(len(local_list))

    north, south, east, west = 0, 100, 0, 300
    for i in range(len(local_list)):
        north = max(north, local_list[i][0])
        south = min(south, local_list[i][0])
        west = min(west, local_list[i][1])
        east = max(east, local_list[i][1])
    north_line = min_lat + (north + 1) * one_km_lat
    south_line = min_lat + south * one_km_lat

    west_line = min_lng + west * one_km_lng
    east_line = min_lng + (east + 1) * one_km_lng
    return north_line, south_line, west_line, east_line, one_km_lat, one_km_lng


def ramble_border(poi_matrix, i, j, sita):
    grid_tuple_consist_of_city_grid = []
    stack = [(i, j)]

    while stack:
        i, j = stack.pop()
        if (i, j) not in grid_tuple_consist_of_city_grid and poi_matrix[i, j] > 10:
            grid_tuple_consist_of_city_grid.append((i, j))

            neighbors = [
                (i + 1, j) if i + 1 != poi_matrix.shape[0] else None,
                (i - 1, j) if i - 1 >= 0 else None,
                (i, j + 1) if j + 1 != poi_matrix.shape[1] else None,
                (i, j - 1) if j - 1 >= 0 else None,
            ]

            for neighbor in neighbors:
                if neighbor is not None:
                    ni, nj = neighbor
                    if (poi_matrix[ni, nj] > poi_matrix[i, j] or
                            (poi_matrix[i, j] - poi_matrix[ni, nj]) / poi_matrix[i, j] < sita):
                        stack.append((ni, nj))

    return grid_tuple_consist_of_city_grid





# def get_border_and_one_km(lat_list, lng_list):
#     lat_unsort = lat_list
#     lng_unsort = lng_list
#     lat_list = sorted(lat_list)
#     lng_list = sorted(lng_list)
#
#     down_lat, up_lat, left_lng, right_lng = 0, 0, 0, 0
#
#     half_lat = lat_list[math.floor(len(lat_list) / 2)]
#     half_lng = lng_list[math.floor(len(lng_list) / 2)]
#     one_km = 1.0 / 111.0
#     one_lng_km = 1.0 / (111.0 * math.cos(math.radians(half_lat)))
#     print(half_lat)
#     print(one_lng_km, "这是横着移动一km有这么多度")
#     print(one_km, "竖着移动一km这么多度")
#
#     #算下边界
#     mid_low_dist = distence((half_lat, max(lng_list)), (min(lat_list), max(lng_list)))
#     poi_number_each_km = []
#
#
#     for i in range(int(mid_low_dist.km)):
#         acount_of_lat = (float(i) * 1.0 / 111.0)
#
#         this_lat = half_lat - acount_of_lat
#
#         theList = [1 for x in range(len(lat_list)) if this_lat > lat_list[x]]
#
#         poi_number_each_km.append(len(theList))
#
#     for i in range(len(poi_number_each_km)):
#         if poi_number_each_km[i] / len(lat_list) * 100 < 10:
#             down_lat = half_lat - one_km * i
#             print("从中间向下", i)
#             break
#     # 算上边界
#     mid_up_dist = distence((half_lat, max(lng_list)), (max(lat_list), max(lng_list)))
#     poi_number_each_km = []
#
#     for i in range(int(mid_up_dist.km)):
#         acount_of_lat = (float(i) * 1.0 / 111.0)
#
#         this_lat = half_lat + acount_of_lat
#
#         theList = [1 for x in range(len(lat_list)) if this_lat < lat_list[x]]
#
#         poi_number_each_km.append(len(theList))
#
#     for i in range(len(poi_number_each_km)):
#         print(poi_number_each_km[i] / len(lat_list) * 100, "%")
#         if poi_number_each_km[i] / len(lat_list) * 100 < 10:
#             up_lat = half_lat + one_km * i
#             print("从中间向上", i)
#             break
#
#     #算左边界
#     mid_left_dist = distence((half_lat, half_lng), (half_lat, min(lng_list)))
#     poi_number_each_km = []
#
#     for i in range(int(mid_left_dist.km)):
#         acount_of_lng = (float(i) * one_lng_km)
#
#         this_lng = half_lng - acount_of_lng
#
#         theList = [1 for x in range(len(lng_list)) if lng_list[x] < this_lng]
#
#         poi_number_each_km.append(len(theList))
#
#     for i in range(len(poi_number_each_km)):
#         print(poi_number_each_km[i] / len(lng_list) * 100, "%")
#         if poi_number_each_km[i] / len(lng_list) * 100 < 10:
#             left_lng = half_lng - one_lng_km * i
#             print("从中间向左", i)
#             break
#
#     mid_right_dist = distence((half_lat, half_lng), (half_lat, max(lng_list)))
#     poi_number_each_km = []
#
#     for i in range(int(mid_right_dist.km)):
#         acount_of_lng = (float(i) * one_lng_km)
#
#         this_lng = half_lng + acount_of_lng
#
#         theList = [1 for x in range(len(lng_list)) if lng_list[x] > this_lng]
#
#         poi_number_each_km.append(len(theList))
#
#     for i in range(len(poi_number_each_km)):
#         print(poi_number_each_km[i] / len(lng_list) * 100, "%")
#         if poi_number_each_km[i] / len(lng_list) * 100 < 10:
#             right_lng = half_lng + one_lng_km * i
#             print("从中间向右", i)
#             break
#
#     return up_lat, down_lat, left_lng, right_lng, one_km, one_lng_km


def get_csvfile(cityname, citypy):
    db = pymysql.connect(host="localhost", user='root', password='uu1003uu', database='city_data', charset='utf8')

    cur = db.cursor()

    statement = 'SELECT `location_lat`, `location_lng`, `tag` FROM `pure_city_poi` where city = "' + cityname + '";'

    cur.execute(statement)

    location_tuple = cur.fetchall()
    location_list = location_tuple



    print('cuda加载完成')
    #得到经纬度数组
    lat_list, lng_list = [], []
    for i in range(len(location_list)):
        lat_list.append(location_list[i][0])
        lng_list.append(location_list[i][1])


    #得到边界和一公里为多少纬度或经度

    up, down, left, right, lat_onekm, lng_onekm = new_get_border(lat_list, lng_list)
    print(up, down, left, right, lat_onekm, lng_onekm, "这个是你想要的数据")
    vertical_dis = (up - down) / lat_onekm
    cross_dis = (right - left) / lng_onekm
    print(vertical_dis, round(vertical_dis))
    print(cross_dis, round(cross_dis))


    tag_type = ['美食', '酒店', '购物', '生活服务', '丽人',
        '旅游景点', '休闲娱乐', '运动健身', '教育培训', '文化传媒',
        '医疗', '汽车服务', '交通设施', '金融',
        '房地产', '公司企业', '政府机构', '出入口', '自然地物', '道路', '门址', '绿地', '行政区划', '地铁线路', '公交线路', '行政地标', '商圈', '铁路', '其他线要素', '水系']

    print(len(tag_type), tag_type)
    whole_data = []
    whole_label = []
    n = 0
    for i in range(round(vertical_dis)):
        this_section_down = down + lat_onekm * i
        this_section_up = down + lat_onekm * (i + 1)
        for j in range(round(cross_dis)):
            this_section_left = left + lng_onekm * j
            this_section_right = left + lng_onekm * (j + 1)

            print("边界值为", this_section_left, this_section_right, this_section_up, this_section_down)

            poi_in_this = [0] * len(tag_type)
            fac_number = 0
            for z in range(len(location_tuple)):
                if this_section_up> location_tuple[z][0] > this_section_down and this_section_right > location_tuple[z][1] > this_section_left:

                    if ';' in location_tuple[z][2]:
                        this_one_type = location_tuple[z][2][0: location_tuple[z][2].index(';')]
                        poi_in_this[tag_type.index(this_one_type)] += 1
                        if '急救中心' in location_tuple[z][2]:
                            fac_number += 1
                    else:
                        this_one_type = location_tuple[z][2]
                        poi_in_this[tag_type.index(this_one_type)] += 1

            poi_in_this.append((this_section_down + this_section_up) / 2)
            poi_in_this.append((this_section_left + this_section_right) / 2)

            whole_data.append(poi_in_this)
            whole_label.append(fac_number)
            print(n+1, "/", round(vertical_dis) * round(cross_dis))
            n += 1


    print(whole_label.count(1))
    with open('../data/' + citypy + ".csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(tag_type + ['纬度', '经度', '急救中心'])
        for i in range(len(whole_data)):
            print(whole_data[i] + [whole_label[i]])
            writer.writerow(whole_data[i] + [whole_label[i]])
    print("小", min(lat_list), "大", max(lat_list))
    print("左", min(lng_list), "右", max(lng_list))

get_csvfile("武汉市", "Wuhan")



# dis_matrix = []
# for i in range(len(location_list)):
#     a_line = []
#     for j in range(len(location_list)):
#         a_line.append(dis(location_list[i], location_list[j]))
#
#     dis_matrix.append(a_line)
#     print("wuhu", i)
#
# dis_matrix = np(dis_matrix)
# print(dis_matrix)
