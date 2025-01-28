import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
print('开始加载模型')

model_path = '/raid/home/zhenghangyang/SIGIR24/bert-base-uncased'

model = BertModel.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
print('加载模型结束')

print('读取文件开始编码')
df = pd.read_csv('item.csv')
from tqdm import tqdm

embeddings = []
for text in tqdm(df['title']):
    inputs = tokenizer(text, return_tensors='pt')

    outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state

    sentence_embedding = last_hidden_state[0][0].detach().numpy()

    embeddings.append(sentence_embedding)

print('正在保存文件。。')
embeddings = np.array(embeddings)
print(embeddings.shape)
np.save('item_text_feature_embeddings_ml-1m_bert.npy', embeddings)
print('程序完成')