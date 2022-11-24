import numpy as np
import pandas as pd
from DecisionTree import DecisionTree
import pickle
from Adaboost import AdaBoost
# # sed = np.loadtxt('dtree.dat', unpack = True)
# print(sed)
from helper import accuracy
def read_Data():
    file=[]
    with open(r"train600.dat",encoding="utf-8") as datFile:
        for i in datFile:
            # if len(i)!=15:
            #     continue
            i=i.split("\n")
            for j in i:
                s=j.split(" ")
                if len(s)!=15:
                    continue
                file.append(j)

    # print(file)
    sentences=[]

    for i in file:
        i=i.split("|")
        sentences.append(i)

    df=pd.DataFrame(sentences,columns=["col1","col2"])
    # d = {'en': 1, 'nl': 0}
    # df["col1"]=df["col1"].map(d)
    #print(df)
    return df

def read_pred_data():
    file = []
    with open(r"train600.dat", encoding="utf-8") as datFile:
        for i in datFile:
            # if len(i)!=15:
            #     continue
            i = i.split("\n")
            for j in i:
                s = j.split(" ")
                if len(s) != 15:
                    continue
                file.append(j)

    # print(file)
    sentences = []

    for i in file:
        i = i.split("|")
        sentences.append(i)

    df = pd.DataFrame(sentences, columns=["col1", "col2"])
    # d = {'en': 1, 'nl': 0}
    # df["col1"]=df["col1"].map(d)
    # print(df)
    return df


def vcratio(s):
    vowels={"a","e","i","o","o"}
    v=0
    c=0
    for i in s:
        if i in vowels:
            v+=1
        else:
            c+=1
    ratio=v/c
    return ratio

def make_flags(df,cutoff_dict):
    for k,v in cutoff_dict.items():
        df[str(k)+"lessthancutoff"]=df[k].squeeze()<v


    return df




def transform_Data(df):

    enwords={'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
             'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
             'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
             'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
             'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or'}

    nlwords={'de', 'en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'een', 'hij',
                'het', 'niet', 'zijn', 'is', 'was', 'op', 'aan', 'met', 'als', 'voor',
                'had', 'er', 'maar', 'om', 'hem', 'dan', 'zou', 'wat', 'mijn', 'men',
                'dit', 'zo', 'door', 'ze', 'zich', 'bij', 'ook', 'tot', 'je', 'mij',
                'uit', 'der', 'daar', 'haar', 'naar', 'heb', 'hoe', 'heeft', 'hebben', 'deze',
                'u', 'nog', 'zal', 'me', 'zij', 'nu', 'ge', 'geen'}


    onlyen=enwords.difference(nlwords)
    onlynl = nlwords.difference(enwords)

    df["english"]=0
    df["dutch"] = 0
    df["avgwordlength"] = 0
    df["largestword"]=0
    df["VCratio"]=0

    for index,row in df.iterrows():
        s=row["col2"]
        words=s.split(" ")
        wordlen=0
        max=0
        for i in words:
            i=i.lower()
            if i in onlyen:
                df.iloc[index, 2]+=1
            if i in onlynl:
                df.iloc[index, 3]+=1
            wordlen+=len(i)
            if len(i)>max:
                max=len(i)
        df.iloc[index, 5] = max
        df.iloc[index,4]=wordlen/15
        df.iloc[index,6]=vcratio(s)


    return df

def driver():
    df=read_Data()
    df=transform_Data(df)
    ## Ever thing is less then
    """
    dutch <2
    english <2
    avg=5.1
    largest word 11
    VC ratio <0.4
    
    """

    cutoff_dict={
        "english":2,
    "dutch":2,
    "avgwordlength":5.1,
        "largestword":11,
        "VCratio":0.4

    }
    # cutoff_dict={
    #     "english":3,
    # "dutch":2,
    # "avgwordlength":5.1,
    #     "largestword":12,
    #     "VCratio":0.35
    #
    # }

    df=make_flags(df,cutoff_dict)
    df = df.sample(frac=1,random_state=42).reset_index(drop=True)

    d = {True: 1, False: 0}
    #print(df.columns)
    df=df[['col1', 'englishlessthancutoff', 'dutchlessthancutoff',
       'avgwordlengthlessthancutoff', 'largestwordlessthancutoff',
       'VCratiolessthancutoff']]


    for i in df.columns[1:]:
        df[i] = df[i].map(d)

    # for i, j in df.iterrows():
    #     print(j)
    df1=df.head(600)
    df2=df.tail(400)
    X=df1[['englishlessthancutoff', 'dutchlessthancutoff',
       'avgwordlengthlessthancutoff', 'largestwordlessthancutoff',
       'VCratiolessthancutoff']]
    y=df1['col1']
    X=X.to_numpy()
    y=y.to_numpy()
    dtree = DecisionTree(level=9)
    root = dtree.train(X, y)
    dtree.travese_tree()
    X2=df2[['englishlessthancutoff', 'dutchlessthancutoff',
       'avgwordlengthlessthancutoff', 'largestwordlessthancutoff',
       'VCratiolessthancutoff']]
    y2=df2['col1']
    X2=X2.to_numpy()
    y2=y2.to_numpy()
    # print(X2[:2])
    # print(y2[:2])
    pred=dtree.pred(X2)
    # print(pred)
    # print(y2)
    print(accuracy(pred,y2))

    ad = AdaBoost(nboost=4)
    ad.train(X,y)
    file = open('model.obj', 'wb')
    pickle.dump(ad, file)
    file.close()

    filehandler = open('model.obj', 'rb')
    ad = pickle.load(filehandler)
    filehandler.close()

    f=ad.pred(X2)
    print(accuracy(f, y2))



driver()


#print(df)
# df1=df[df.col1==1]
# print(df1.describe())
#
#
# df1=df[df.col1==0]
# print(df1.describe())




# df=pd.read_csv("dtree.dat",header=0)
# print(df)
#
# cols=["col"+str(i) for i in range(len(file[0]))]
#
#
# df=pd.DataFrame(file, columns =cols)
#
# d = {'True': 1, 'False': 0}
# for i in cols[:-1]:
#     df[i] = df[i].map(d)
# df=df.to_numpy()
# #print(df)
#
# dtree=DecisionTree(level=2)
#
# data=df[:,:-1]
# labels=df[:,-1]
# root=dtree.train(data,labels)
# file = open('model.obj', 'wb')
# pickle.dump(dtree, file)
# file.close()
#
# # with open('model.obj','rb') as file_object:
# #     dtree = file_object.read()
# filehandler = open('model.obj', 'rb')
# dtree = pickle.load(filehandler)
# filehandler.close()
# print(dtree)
#
# dtree.travese_tree(root)
