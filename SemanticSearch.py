import json
import numpy as np
import pandas as pd
from model.utils import pytorch_cos_sim
from data.dataloader import convert_to_tensor, example_model_setting


def main():
    model_name = 'klue/bert-base'
    model_ckpt = '../Checkpoint/KoSimCSE/kosimcse-klue-bert-base.pt'
    model, tokenizer, device = example_model_setting(model_ckpt, model_name)

    # Corpus with example sentences
    em_corpus = pd.read_csv('test_corpus.csv')
    corpus = em_corpus['main_text'].tolist()
    inputs_corpus = convert_to_tensor(corpus, tokenizer, device)

    corpus_embeddings = model.encode(inputs_corpus, device)
    em_se_numpy = corpus_embeddings.detach().numpy()
    
    for i in range(em_se_numpy.shape[0]):
        em_corpus.loc[i,'embedding'] = json.dumps(em_se_numpy[i,:].tolist())

    print(em_corpus)

    # Query sentences:
    queries = ['스포츠 대회 관련 기사 찾아줘',
               '남현희 관련 기사 추천해줘',
               '최근 마약 사건관련 기사 추천해줘']

    # Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
    top_k = 3
    for query in queries:
        query_embedding = model.encode(convert_to_tensor([query], tokenizer, device), device)
        
        cos_scores = pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu().detach().numpy()

        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for idx in top_results[0:top_k]:
            print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))


if __name__ == '__main__':
    main()
