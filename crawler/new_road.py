CUDA_VISIBLE_DEVICES=0
import pymysql
import requests
import json
from bs4 import BeautifulSoup
def take_poi(city_name, city_py):
    url = "https://poi.mapbar.com/" + city_py
    res = requests.get(url)
    html = res.content.decode('utf-8')
    soup = BeautifulSoup(html, 'lxml')
    again_and_again = 0

    db = pymysql.connect(host="localhost", user='root', password='uu1003uu', database='city_data', charset='utf8')
    cur = db.cursor()
    stat1 = 'SELECT name FROM `word_for_baidu_poi` where city like' + ' "' + city_name + '"' + ';'
    cur.execute(stat1)
    name_list = [k[0] for k in cur.fetchall()]
    statement = "INSERT INTO `word_for_baidu_POI`(`name`, `type`, city) values(%s, %s, %s)"

    data = []

    # for i in range(1, 231):
    #     print(soup.findAll('a', target='')[i])

    # res1 = requests.get(soup.findAll('a', target='')[6].get('href'))
    # html1 = res1.content.decode('utf-8')
    # soup1 = BeautifulSoup(html1, 'lxml')

    # for i in range(len(soup1.findAll('a', id="", target=''))):
    #     print(soup1.findAll('a', id="", target='')[i])


    #其他公司企业 228
    all_type = soup.findAll('a', target='')
    for i in range(1, 300):
        if all_type[i - 1].text == '其他公司企业':
            break
        print(all_type[i].text, i)
        print(again_and_again)
        res1 = requests.get(all_type[i].get('href'))
        html1 = res1.content.decode('utf-8')
        soup1 = BeautifulSoup(html1, 'lxml')
        poi_list_in_this_type = soup1.findAll('a', id="", target='')[2: -1]
        for j in range(len(poi_list_in_this_type)):
            if poi_list_in_this_type[j].text in name_list:
                again_and_again += 1
                continue
            one_data_list = [poi_list_in_this_type[j].text, all_type[i].text, city_name]
            cur.execute(statement, one_data_list)
            db.commit()

    print("****************************************************")
    db.close()

city_name_list = ['蚌埠市', '鄂尔多斯市', '吕梁']
city_py_list = ['bengbu', 'eerduosi', 'lvliang']

for i in range(len(city_py_list)):
    take_poi(city_name_list[i], city_py_list[i])