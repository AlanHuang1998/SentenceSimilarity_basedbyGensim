# SentenceSimilarity_basedbyGensim

### 專案概覽
這個程式庫示範如何用 Gensim 的 Word2Vec 與 Jieba 分詞比較中文句子相似度。流程包括：先將資料庫中的句子切詞並訓練詞向量模型，再將任意句子轉為向量後計算多種距離或相似度

### 主要程式
- Trainmodel()：讀取 句子資料庫.txt，使用 Jieba 切詞，生成 word2vec.model 供後續比較使用
- Usemodel()：載入模型，將新句子切詞後平均其詞向量，並輸出歐式距離、Cosine、Jaccard、Hamming、Correlation 以及 Jaro-Winkler 等多種相似度指標
- 資料庫示例包含與空氣品質、健康相關的中文句子，作為訓練與測試基礎

### 入門重點
- 環境依賴：需安裝 gensim、jieba、scipy 及 pyjarowinkler 等套件。
- 分詞與向量化：Jieba 可自訂字典以提升斷詞品質；Word2Vec 模型會以 size=250 的向量表示每個詞。
- 句子向量：將句中所有詞向量取平均，得到句子的整體表示，再以多種距離衡量相似度。

### 下一步學習建議
- 優化分詞：嘗試自訂 Jieba 字典或使用 TF-IDF 權重，以改善關鍵詞的處理。
- 改進向量表示：探索加權平均、Doc2Vec 或使用預訓練的深度模型（如 BERT）提升句子表示能力。
- 評估與應用：深入了解各種距離度量的特性，並測試在實際情境（如搜尋、推薦或問答）中的效果。
