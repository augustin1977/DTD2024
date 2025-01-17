import os
import json
from collections import Counter
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel,AutoTokenizer
from funcoes_auxiliares import *
class AI:
    def __init__(self, df):
        print("#### Iniciando o treinamento ####")
        self.pasta=os.getcwd()
        self.ai_model="neuralmind/bert-large-portuguese-cased"
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(self.ai_model)
        self.model = AutoModel.from_pretrained(self.ai_model)
        self.vectorizer = TfidfVectorizer()
        self.projetos_selecionados=[]
        if os.path.exists('tokens.json'):
            print("Carregando tokens existentes...")
            with open('tokens.json', 'r', encoding='utf-8') as f:
                tokens = json.load(f)
            self.df['Tokens'] = tokens
        else:
            print("Gerando tokens...")
            self.df['Tokens'] = self.df['Resumo_limpo'].apply(lambda x: self.tokenizer.tokenize(x))
            with open('tokens.json', 'w', encoding='utf-8') as f:
                json.dump(self.df['Tokens'].tolist(), f, ensure_ascii=False, indent=4)

        if os.path.exists('embeddings.npy'):
            print("Carregando embeddings existentes...")
            embeddings = np.load('embeddings.npy')
            self.df['Embedding'] = list(embeddings)
        else:
            print("Não encontrado arquivo de Embeddings")
            self.gerar_embeddings()
            embeddings = np.vstack(self.df['Embedding'].values)
            np.save('embeddings.npy', embeddings)

        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Resumo_limpo'])

    def gerar_embedding(self, texto):
        inputs = self.tokenizer(texto, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding

    def gerar_embeddings(self):
        print("Gerando Embedings... e fazendo fine training")
        self.df['Embedding'] = self.df['Resumo_limpo'].apply(self.gerar_embedding)
        print("Embeddings gerados com sucesso!")

    def buscar_resposta(self, pergunta, top_n=3):
        print("Buscando projetos com essa temática....")
        
        # Gera o embedding da pergunta
        embedding_pergunta = self.gerar_embedding(pergunta)
        
        # Converte a coluna de embeddings para uma matriz numpy
        embeddings = np.vstack(self.df['Embedding'].values)
        
        # Calcula a similaridade entre a pergunta e todos os resumos
        similaridades = cosine_similarity([embedding_pergunta], embeddings)[0]
        
        # Encontra os índices dos top N projetos mais similares
        indices = np.argsort(similaridades)[-top_n:][::-1]  # Ordena do maior para o menor
        
        # Recupera os top N projetos
        projetos_selecionados = self.df.iloc[indices]
        
        # Soma as palavras-chave dos projetos selecionados
        palavras_chave_somadas = Counter()
        for palavras_chave in projetos_selecionados['Palavras_chave']:
            palavras_chave_somadas.update(palavras_chave)
        
        # Retorna os projetos selecionados e as palavras-chave somadas
        self.projetos_selecionados=projetos_selecionados
        return palavras_chave_somadas

    def recomendar_pesquisadores(self, palavra_chave=[], top_n=10):
        self.participantes = {}
        print("Buscando pesquisadores....")
        for _, row in self.df.iterrows():
            for pc in palavra_chave:
                if pc in row["Palavras_chave"]:
                    for participante in row["Equipe"]:
                        pontos=calcula_pontos(row["Ano"])
                        # ~ print(participante,row["Ano"],pontos)
                        if participante in self.participantes:
                            self.participantes[participante] += pontos
                            
                        else:
                            self.participantes[participante] = pontos
        self.pesquisadores_ordenados = sorted(self.participantes.items(), key=lambda x: x[1], reverse=True)
        print(self.pesquisadores_ordenados[:top_n*2])
        return [pesquisador for pesquisador, _ in self.pesquisadores_ordenados[:top_n]]


