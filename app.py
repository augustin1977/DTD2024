import chainlit as cl
from ai import *
from importando_dados import *
from funcoes_auxiliares import *

root=os.getcwd()
endereco=os.path.join(root,"Extração de dados","Relatórios Biblioteca")
planilha=Importacao(endereco)
planilha.importa_dados()
planilha.cria_json()
df=planilha.get_df()
ai=AI(df)

@cl.on_chat_start
async def on_chat_start():
    prompt="Posso te ugerir uma equipe para um projeto, mas preciso que descreva o tipo de problema que você está querendo resolver:"
    await cl.Message(content=prompt).send()
    
@cl.on_message
async def main(message: cl.Message):
    
    palavras_chaves= ai.buscar_resposta(message.content, top_n=6)
    reposta=ai.recomendar_pesquisadores(palavra_chave=palavras_chaves,top_n=8)
    # reposta.sort()
    pesquisadores = ", ".join(reposta)+"."
    await cl.Message(content=f"Os pesquisadores que podem ajudar neste tipo de projeto são: {pesquisadores}").send()