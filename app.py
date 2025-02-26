import chainlit as cl
from Extraindo_metadados_xlsx import *
import os
from huggingface_hub import login
from key import huggings_TOKEN

# Criando os objeto


login(token=huggings_TOKEN)
root=os.getcwd()
print(root)
endereco=os.path.join(root,"Extração de dados","Relatórios Biblioteca")
print(endereco)
planilha=Importacao(endereco)
planilha.importa_dados()
planilha.cria_json()
df=planilha.get_df()
print("Carregando Pesquisadores Ativos")
arquivo_pesquisadores_ativos=os.path.join(root,"Extração de dados","pesquisadores_ativos","pesquisadores_ativos.xlsx")
pesquisadores_ativos=carrega_pesquisadores_ativos(arquivo_pesquisadores_ativos)

ai=AI(df)


@cl.on_chat_start
async def on_chat_start():
    prompt="Posso te sugerir uma equipe para um projeto, mas preciso que descreva o tipo de problema que você está querendo resolver:"
    await cl.Message(content=prompt).send()
    # respodendo
@cl.on_message
async def main(message: cl.Message):
    pergunta= message.content
    if len(pergunta)<20:
        await cl.Message(content=f"Pergunta muito curta, tente explicar mais....").send()
        return
    resposta=ai.buscar_resposta(pergunta, top_n=10,)
    resposta_ai="Os projetos que encontrei similaridade foram:\n"
    for _, projeto in ai.projetos_selecionados.iterrows():
        resposta_ai+=f"Título: {projeto['Titulo']}"+"\n"
        
    pesquisadores_ordenados=ai.recomendar_pesquisadores(lista_pesquisadores=resposta,
                                                    top_n=6,
                                                    pesquisadores_ativos=pesquisadores_ativos)
    pesquisadores="\n".join(pesquisadores_ordenados)
    resposta_ai+=f"As pessoas que podem te ajudar são:\n{pesquisadores}"
    await cl.Message(content=resposta_ai).send()

