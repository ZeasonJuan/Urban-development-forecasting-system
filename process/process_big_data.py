import pymysql
def insertdata(cityname):
    db = pymysql.connect(host="localhost", user='root', password='uu1003uu', database='city_data', charset='utf8')
    cur = db.cursor()

    statement1 = "SELECT `city`, `POI_name`, `location_lat`, `location_lng`, `uid`, `tag` FROM `city_POI_copy1` where city = " + "'" +cityname + "'"
    statement2 = "SELECT `city`, `POI_name`, `location_lat`, `location_lng`, `uid`, `tag` FROM `city_POI` where city = " + "'" +cityname + "'"

    cur.execute(statement2)
    s = cur.fetchall()

    cur.execute(statement1)

    s1 = cur.fetchall()
    s2 = s + s1
    cur.execute("SELECT `city`, `POI_name`, `location_lat`, `location_lng`, `uid`, `tag` FROM `pure_city_poi`")
    s3 = cur.fetchall()
    uid_set = set()
    for i in range(len(s3)):
        uid_set.add(s3[i][4])
    state_3 = "INSERT INTO `pure_city_poi`(`city`, `POI_name`, `location_lat`, `location_lng`, `uid`, `tag`) VALUES(%s, %s, %s, %s, %s, %s)"
    for i in range(949600, len(s2)):
        if s2[i][4] not in uid_set:
            uid_set.add(s2[i][4])
            cur.execute(state_3, s2[i])
            db.commit()
        if i % 100 == 0:
            print(i)
    db.close()

city_list = ['北京市', '上海市', '深圳市', '广州市', '合肥市', '南京市', '宁德市', '厦门市', '武汉市', '长沙市', '天津市', '杭州市', '西安市', '成都市', '菏泽市']

insertdata("杭州市")
