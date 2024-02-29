import pymysql
import requests
import json
import urllib.request
import string
import difflib
import datetime

proxies={
'http': 'http://127.0.0.1:4780',
'https': 'http://127.0.0.1:4780'  # https -> http
}
def string_sim(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

def get_json(particular_word):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16'
    }


    params = {
        'query': particular_word,                #检索关键字
        #*****************************************************************************************************************************
        'region': '北京市',              #检索行政区划区域
        'city_limit': True,
        'output': 'json',               #输出格式为json
        'scope': '2',                   #检索结果详细程度。取值为1 或空，则返回基本信息；取值为2，返回检索POI详细信息
        'page_size': 20,                #单次召回POI数量，默认为10条记录，最大返回20条。
        'page_num': 0,                  #分页页码，默认为0,0代表第一页，1代表第二页，以此类推。
        'ak': 'oSiqSuES2L7fU1TXqClMhOOv0eTDGpTr'
    }


    res = requests.get("https://api.map.baidu.com/place/v2/search", headers=headers, params=params)
    content = res.text
    decodejson = json.loads(content, strict=False)  #将已编码的 JSON 字符串解码为 Python 对象，就是python解码json对象
    return decodejson

db = pymysql.connect(host="localhost", user='root', password='uu1003uu', database='city_data', charset='utf8')

cur = db.cursor()
# *****************************************************************************************************************************
statement = "SELECT `name`, `type`, `city` FROM `word_for_baidu_poi` where `city` = '北京市';"

cur.execute(statement)
adata_list = cur.fetchall()
print('已导入关键词')
db.close()

db = pymysql.connect(host="localhost", user='root', password='uu1003uu', database='city_data', charset='utf8')

cur = db.cursor()

statement = "SELECT * FROM `city_poi`;"


cur.execute(statement)
in_data = cur.fetchall()
print('已有数据库检索完成')
db.close()
list_data_in = []
uid_list_in_database = []
# for i in range(len(in_data)):
#     list_data_in.append(in_data[i][2])
#     uid_list_in_database.append(in_data[i][5])

temp_list_hoard_data = []
uid_temp_list = []

db = pymysql.connect(host="localhost", user='root', password='uu1003uu', database='city_data', charset='utf8')

cur = db.cursor()
statement = "INSERT INTO `city_POI_copy1`(`city`, `POI_name`, `location_lat`, `location_lng`, `uid`, `tag`) VALUES(%s, %s, %s, %s, %s, %s)"

#3/6/ 我现在是要关机了，查到26000
jump_time_like = 0
jump_time_has = 0


batch_poi_message = []

for i in range(328019, len(adata_list)):
    #为了减少额度使用

    # if adata_list[i][1] != '急救中心' and i % 3 != 0:
    #     continue
    #为了减少额度使用 以及加快检索速度
    if string_sim(adata_list[i][0], adata_list[i - 1][0]) > 0.5 and i > 0:
        print(adata_list[i][0], '和', adata_list[i - 1][0], "相似度高达70%，不查了", jump_time_like)
        jump_time_like += 1
        continue


    #已经有的就不查了为了减少次数   节约额度
    # if adata_list[i][0] in list_data_in:
    #     print(adata_list[i][0], '数据库中已有', jump_time_has)
    #     jump_time_has += 1
    #     continue



    decoded = get_json(adata_list[i][0])




    #为了防止因为没有results导致的程序停止
    if len(decoded['results']) != 0:
        #把能拿回来的都拿回来
        for each_result in decoded['results']:
            #防止因为没有某些字段导致程序崩溃
            if 'city' in each_result and 'tag' in each_result['detail_info']:
                # if each_result['uid'] in uid_list_in_database:
                #     #print("已进行检索到的", each_result['name'], "已在数据库中,这是该次检索的第", each_result, "个")
                #     continue

                one_POI_message = [each_result['city'], each_result['name'],
                                   each_result['location']['lat'],
                                   each_result['location']['lng'], each_result['uid'],
                                   each_result['detail_info']['tag']]
                batch_poi_message.append(one_POI_message)

                # #加入列表(name)方便开局的时候查询，以节约次数
                # list_data_in.append(each_result['name'])
                #加入列表(uid)方便已查询后再次查询，以防止重复
                # uid_list_in_database.append(each_result['uid'])

        # 为了显示进度
    if i % 29 == 0 or i == len(adata_list) - 1:
        print('现在进行到', i, '次搜索')
        cur.executemany(statement, batch_poi_message)
        db.commit()
        batch_poi_message.clear()
    #         else:
    #             print('少了一个或多个字段， 这是这次检索的第', each_result, "个")
    # else:
    #     print(decoded)


print(len(adata_list), '这个是最后爬到的位置')
print(datetime.datetime.now())
db.close()
print("因为相似度跳了", jump_time_like, "下")
print("因为已经有了跳了", jump_time_has, '下')

print('节省了', jump_time_has + jump_time_like, "次检索机会")

