# 安裝套件
from gensim.models import word2vec
import numpy as np
import jieba
import scipy
from pyjarowinkler import distance
#jieba.set_dictionary('jieba字典.txt') # 由於Jieba支援替換字典，因此可以使用自製的字典，恕不提供

def Trainmodel():
    # 讀取文庫句子
    with open('句子資料庫.txt', 'r', encoding='UTF-8')as f :
        SentenceDatabase = f.read().split()
    # 將句子逐一斷詞並儲存
    with open('分詞後的句子.txt', 'w', encoding='UTF-8') as d:
        for sentence in SentenceDatabase:
            jword = jieba.cut(sentence, cut_all=False)
            d.write(" ".join(jword) + ' ')
    # 把詞轉換成詞向量並儲存
    sentences = word2vec.LineSentence("分詞後的句子.txt")
    model = word2vec.Word2Vec(sentences, vector_size=250, min_count=1)
    model.save('word2vec.model')

def Usemodel():
    model = word2vec.Word2Vec.load('word2vec.model') # 讀取model
    # 句子1的轉換
    InputString = '高血脂的定義'
    jword = jieba.cut(InputString, cut_all=False)
    veclist = []
    for word in jword:
        veclist.append(model[word])
    metrixlist = np.array(veclist)
    Metrix1 = np.mean(metrixlist, axis = 0) # 計算句子的向量
    
    # 句子2的轉換
    InsideString = '高血壓定義'
    jword2 = jieba.cut(InsideString, cut_all=False)
    veclist2 = []
    for word in jword2:
        veclist2.append(model[word])
    metrixlist = np.array(veclist2)
    Metrix2 = np.mean(metrixlist, axis = 0) # 計算句子的向量

    # 以下是各種比較方式，比較準確的有Cosine similarlity，jarodistance，相關距離
    # 歐式距離
    Ohdist = np.linalg.norm(Metrix1-Metrix2)
    print("歐式距離: ", Ohdist*100)

    # Cosine similarlity
    Cosinedist = scipy.spatial.distance.cosine(Metrix1, Metrix2)
    print("Cosine_Similarity: ", 1-Cosinedist)

    # 傑卡德距離
    Jaccdistance = scipy.spatial.distance.jaccard(Metrix1, Metrix2)
    print("Jacc_distance: ", 1-Jaccdistance)

    # 漢明距離
    Hammdistance = scipy.spatial.distance.hamming(Metrix1, Metrix2)
    print("Hamm_distance: ", 1-Hammdistance)

    # 相關距離
    Corrdistance = scipy.spatial.distance.correlation(Metrix1, Metrix2)
    print("Corr_distance: ", 1-Corrdistance)

    # jarodistance
    Jarodistance = distance.get_jaro_distance(InputString, InsideString, winkler=True, scaling=0.1)
    print("Jaro_distance: ", Jarodistance)


if __name__ == "__main__":
    Trainmodel()
    Usemodel()
    
