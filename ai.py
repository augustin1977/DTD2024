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
        """Inicializa a AI baixando o modelo caso não estja na pasta modelo
        e cria a lista de tokes e de embbedings caso não estejam na pasta modelo"""
        print("#### Iniciando o treinamento ####")
        pasta=os.getcwd()
        self.local=os.path.join(pasta,"modelo")
        self.ai_model="neuralmind/bert-large-portuguese-cased"
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(self.ai_model,cache_dir=self.local)
        self.model = AutoModel.from_pretrained(self.ai_model,cache_dir=self.local)
        self.vectorizer = TfidfVectorizer()
        self.projetos_selecionados=[]
        self.token_file= os.path.join(self.local,'tokens.json')
        self.embeddings_file= os.path.join(self.local,'embeddings.npy')
        if os.path.exists(self.token_file):
            print("Carregando tokens existentes...")
            with open(self.token_file, 'r', encoding='utf-8') as f:
                tokens = json.load(f)
            self.df['Tokens'] = tokens
        else:
            print("Gerando tokens...")
            self.df['Tokens'] = self.df['Resumo_limpo'].apply(lambda x: self.tokenizer.tokenize(x))
            with open(self.token_file, 'w', encoding='utf-8') as f:
                json.dump(self.df['Tokens'].tolist(), f, ensure_ascii=False, indent=4)

        if os.path.exists(self.embeddings_file):
            print("Carregando embeddings existentes...")
            embeddings = np.load(self.embeddings_file)
            self.df['Embedding'] = list(embeddings)
        else:
            print("Não encontrado arquivo de Embeddings")
            self.gerar_embeddings()
            embeddings = np.vstack(self.df['Embedding'].values)
            np.save(self.embeddings_file, embeddings)

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
        """Recebe um pergunta e um inteiro(top_n), aplica essa pergunta no LLM e busca projetos cujo resumo tenha similaridade com a pergunta,
        eetorna uma lista das palavras dos top_n projetos mais similares"""
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
        """Recebe umna lista de palavras chaves e busca todos os pesquisadores que ja fizeram parte de projetos 
        com pelo menos uma dessas palavras chaves """
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


