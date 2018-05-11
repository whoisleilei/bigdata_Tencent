#coding:utf-8
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
"""
1.弄三个csv的demo，用来debug用
2.用户特征太大了，没办法直接加在成pandas，要做格式转换，弄成字典之后再弄成dataframe
3.特征拼接，把所有的拼到一起（这里train和test的一起弄，train的-1弄成0，然后test的标签先都打成-1，这样子到时候好区分）
4.缺失值填充成“-1”，字符型的-1
4.应该弄个数据压缩
5.one-hot编码不用弄，直接用lightgbm就行
6.这是一个二分类问题（binary）
"""

"""
广告特征（ad_feature）:
aid             int64
advertiserId    int64
campaignId      int64
creativeId      int64
creativeSize    int64
adCategoryId    int64
productId       int64
productType     int64



用户特征（user-feature）：
['LBS', 'age', 'appIdAction', 'appIdInstall', 'carrier', 'consumptionAbility', 'ct', 'education', 'gender', 'house', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'marriageStatus', 'os', 'topic1', 'topic2', 'topic3', 'uid']
LBS                   float64
age                     int64
appIdAction            object
appIdInstall           object
carrier                 int64
consumptionAbility      int64
ct                     object
education               int64
gender                  int64
house                 float64
interest1              object
interest2              object
interest3              object
interest4              object
interest5              object
kw1                    object
kw2                    object
kw3                    object
marriageStatus         object
os                     object
topic1                 object
topic2                 object
topic3                 object
uid                     int64
dtype: object


"""
data_path="../data/"

#离散型的特征，数一数有多少个值，每个值有多少个
def discrete_count(feature_col):     #把那个特征列传进来
    print (feature_col.value_counts())



#数据有点大，这个用来读几行数据看看长啥样
#我想知道的，有哪些列，没列是离散型还是连续型特征，离散型有多少个取值
def View_ad_feature_data():     #把data（dataframe型）传进来
    ad_feature = pd.read_csv(data_path + "adFeature.csv")
    print (ad_feature)

    #这个能输出都有那些列
    columns=list(ad_feature.columns)
    print (columns)

    #这个能输出都有哪些列，这些列的数据类型都是啥
    datatype=ad_feature.dtypes
    print (datatype)

    #这个可以看离散的每一列都有哪些值，然后这些值有多少个。（注意，这个ad_feature恰好全是int64的）
    for feature in columns:
        discrete_count(ad_feature[feature])

#用来生成用户特征的csv
def user_feature_csv_generate():
    # 搬砖,就是先把user_feature读成字典，然后再弄成dataframe
    userFeature_data = []
    with open(data_path + "userFeature.data", 'r') as f:
        #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        for i, line in enumerate(f):
            #strip用来移除前后的空格
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 100000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        #print (user_feature.iloc[0:50,:])
        user_feature.to_csv(data_path + "userFeature.csv", index=False)




#取几行看看用户特征啥德行,每次的结果都记下来，否则太大了不好弄
def View_user_feature_data():
    print ("hello")
    user_feature = pd.read_csv(data_path + "userFeature.csv",nrows=50)
    print (user_feature)

    # 通过这样的代码测试，得到这个apply(int)的作用就是把各种各样的数据类型转成int
    print (user_feature["house"].fillna("-1",inplace=True))
    #print (user_feature["house"].apply(int))

    #然后这里就很神奇了。这个apply(int)的功能，是把所有的不是int都转成int的，因为LabelEncoder只能弄都是int的。
    #之前house缺失值填充的是字符的"-1"，所以这里要用apply(int)转一下。所以觉得那个人的代码缺失值完全没必要填充字符的"-1"啊，直接填充数字的-1不就可以了......
    #但是那个人的代码几个版本这个问题都没有改，为什么？
    user_feature["house"] = LabelEncoder().fit_transform(user_feature["house"].apply(int))
    print (user_feature["house"])

    #这个apply(int)还是有用，因为可能不止int和str混，还有object，想labelEncoder也得转成int
    #问题又来了，能不能不用labelEncoder(不过这个labelEncoder方便，labelEncoder的作用，不论这个类别的数字编码多大多炫酷，统统打回原型从0开始数......本质上就是个类别，就让他们变成只有0，1，2，3......这样从头开始数的基础的类别)
    user_feature["interest5"].fillna(-1, inplace=True)
    print(user_feature["interest5"])
    user_feature["interest5"] = LabelEncoder().fit_transform(user_feature["interest5"])

    # 这个能输出都有那些列
    columns = list(user_feature.columns)
    print(columns)

    # 这个能输出都有哪些列，这些列的数据类型都是啥
    datatype = user_feature.dtypes
    print(datatype)

    # 这个可以看离散的每一列都有哪些值，然后这些值有多少个。
    # for feature in columns:
    #     discrete_count(ad_feature[feature])

#(写一堆函数是怕中间出问题就不能弄了，因为这个数据量太大)
#把广告特征，用户特征，标签都合起来，并且生成csv。train和test放一起合方便，要不到时候test也得合，太费劲了......
def train_pred_feature_concat():
    train_data=pd.read_csv(data_path+"train.csv")
    #print (train_data)
    print (1)
    test_data=pd.read_csv(data_path+"test1.csv")
    print (2)
    #print (test_data)
    #train的-1弄成0，然后test的标签先都打成-1
    train_data.loc[train_data["label"]==-1,"label"]=0
    test_data["label"]=-1
    data=pd.concat([train_data,test_data])   #这里的concat是竖着的concat，直接竖着接起来的
    print (3)
    #print (data)
    ad_feature=pd.read_csv(data_path+"adFeature.csv")
    user_feature=pd.read_csv(data_path+"userFeature.csv")
    data=pd.merge(data,ad_feature,on='aid',how='left')     #left:参与合并左侧的dataframe,就是都合到data里
    data=pd.merge(data,user_feature,on='uid',how='left')
    print (4)
    data.to_csv(data_path+"mergedData.csv",index=False)
    print (data.loc[0:50,:])
    return data



def fillNA_data():
    mergedData=pd.read_csv(data_path+"mergedData.csv")
    mergedData.fillna("-1",inplace=True)    #缺失值填充成字符的-1，虽然我并不是很知道为啥......先有个区分吧
    print(mergedData.loc[0:50, :])
    mergedData.to_csv(data_path+"mergedDataFillna.csv",index=False)


#查看数据前50行，看样子这种
def view_data():
    temp=pd.read_csv(data_path+"mergedDataFillna.csv",nrows=50)
    print (temp)



if __name__=="__main__":
    print ("hello")
    disperse_feature=[""]
    View_ad_feature_data()
    #user_feature_csv_generate()
    View_user_feature_data()
    train_pred_feature_concat()
    fillNA_data()
    view_data()