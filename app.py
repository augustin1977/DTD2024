import chainlit as cl
import asyncio
from Extraindo_metadados_xlsx import *
import os
from huggingface_hub import login
from key import huggings_TOKEN

# Criando os objeto


login(token=huggings_TOKEN)
root=os.getcwd()

endereco=os.path.join(root,"Extração de dados","Relatórios Biblioteca")
print("Preparando dados....")
planilha=Importacao(endereco)
if not (os.path.exists('tokens.json') and os.path.exists('embeddings.npy')):
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
    resposta_ai="Estou buscando os projetos:\nOs projetos que encontrei similaridade foram:\n"
    for _, projeto in ai.projetos_selecionados.iterrows():
        resposta_ai+=f"Título: {projeto['Titulo']} Ano:{projeto['Ano']}"+"\n"
        
    pesquisadores_ordenados=ai.recomendar_pesquisadores(lista_pesquisadores=resposta,
                                                    top_n=10,
                                                    pesquisadores_ativos=pesquisadores_ativos)
    pesquisadores=[]
    for pesquisador in pesquisadores_ordenados:
        pesquisadores.append(f"Pesquisador: {pesquisador[0]} - Pontuação: {pesquisador[1]:.2f}")
    
    pesquisadores="\n".join(pesquisadores)
    resposta_ai+=f"##########################\nAs pessoas que podem te ajudar são:\n{pesquisadores}"
    ms=cl.Message(content="")
    await ms.send()
    for line in resposta_ai.strip().split("\n"):
        await asyncio.sleep(0.7)  # Pequeno atraso entre cada linha
        if "Os projetos que encontrei similaridade foram:" in line or "similaridade" in line:  
            await asyncio.sleep(2)
        elif "###" in line:
            await asyncio.sleep(3)
        await ms.stream_token(line + "\n")
    await ms.stream_token("Espero que as informações tenha sido uteis, se quiser pesquisar mais, é só avisar.👍🏻\n")
