import tabula # busca tabela no pdf
import pdfplumber # busca texto corrido no pdf
import pandas as pd # pandas para analise de tabelas
import os
from datetime import datetime
import json
import time
import random
import re

def testa_data(texto):
    info=texto.split(' ')
    if info[0].isdigit():
        return texto
        
    return False
        

def busca_metadados(arquivo):
	texto=pdfplumber.open(arquivo) # carrega o arquivo
	metadados={}#texto.metadata # cria os metadados
	indice=texto.pages[0].extract_text().split("\n") # extrai o texto da primeira pagina e busca as informações de indice, Titulo,cliente, laboratório, unidade entre outros   
	metadados["Unidade"]=None
	for i in range(len(indice)): # percorre toda a pagina
	 
		if 'CLIENTE' in indice[i]:
			metadados['Cliente']=indice[i+1].title() # seleciona cliente
		
		elif (('UNIDADE RESPONSÁVEL' in indice[i] or 
				'RESPONSÁVEL' in indice[i] or 
				'UNIDADES RESPONSÁVEIS' in indice[i] or 
				'UNIDADE DE NEGÓCIO RESPONSÁVEL' in indice[i]) and
				not metadados["Unidade"]): # seleciona unidade e laboratório
			metadados['Unidade']=indice[i+1].upper()
			metadados['Laboratorio']=indice[i+2].upper()
		elif testa_data(indice[i]): # chama a função que localiza a data do documento
			metadados['Data']=indice[i] # seleciona data
			# copia o titulo que sempre vem a seguir da data até a palavra relatório ou final ou parcial ou cliente serem encontradas
			a=1
			titulo=""
			# ~ while(not ("final" in indice[i+a].lower() or 
						# ~ "relatório" in indice[i+a].lower() or 
						# ~ "parcial" in indice[i+a].lower() or 
						# ~ "cliente" in indice[i+a].lower())):
			while(not ("cliente" in indice[i+a].lower())):
				titulo=titulo+indice[i+a]+" "
				a+=1
			titulo = re.sub(r"(-|Relatório|–|:).*", "", titulo).strip().capitalize() # exclui os termos relatório, - e : da string
			metadados['Titulo']=titulo # Atribui o titulo aos metadados
	if metadados["Unidade"] and metadados["Laboratorio"]:
		if "laboratório".upper() in metadados["Unidade"].upper() or "laboratorio".upper() in metadados["Unidade"].upper(): # Se a palavra laboratório estiver na unidade
			metadados["Unidade"],metadados["Laboratorio"]=metadados["Laboratorio"],metadados["Unidade"] # troca o contudo de laboratório com unidade
	paginas=texto.pages
	metadados['Numero_paginas']=len(paginas)+1 # coloca o numero de paginas nos metadados
	
	a=1
	palavraschave=[] 
	while(palavraschave==[] and a<10):
		indice=texto.pages[a].extract_text().split("\n") # extrai od texto as palavras chaves da segunda pagina em diante até encontra-las mas não vai alem da 10 pagina
		for i in range(len(indice)):
			if "palavras chave".upper() in indice[i].upper() or "palavras-chave".upper() in indice[i].upper() or "palavras - chave".upper() in indice[i].upper():
				if len(indice[i].strip())>20: # se o comprimento da linha escrito palavras chaves for grande (>20) entãos as palavras chaves estão na mesma linha do titulo
					try:
						palavraschave= indice[i].strip()+" "+indice[i+1].replace(".","").strip()# Remove o ultimo caracter
					except:
						palavraschave= indice[i].replace(".","").strip()# Remove o ultimo caracter
				else:
					try:
						palavraschave= indice[i+1].strip()+" "+indice[i+2].replace(".","").strip()# Remove o ultimo caracter 
					except:
						palavraschave= indice[i+1].replace(".","").strip()# Remove o ultimo caracter 
				if ":" in palavraschave:
					palavraschave=palavraschave.split(":")[1] #Separa a parte que em depois do : no string
				if "," in palavraschave:
					palavraschave=palavraschave.split(",") # Se o sepadados for a virgula, usa ele para criar a lista de palavras
				elif(";" in palavraschave):
					palavraschave=palavraschave.split(";") #Se o sepadados for o ponto e virgula, usa ele para criar a lista de palavras
				else:
					palavraschave=[] # caso não for nenhum dos dois, retorna a lista vazia
		a+=1
	palavras_chave=[]
	for p in palavraschave:
		palavras_chave.append(p.strip().capitalize())
	metadados["palavras_chave"]=palavras_chave
	
	
	metadados['Equipe']=[] # Cria a lista de autores / equipe
	for i in range(len(paginas)-1,-1,-1): # percorre todas as paginas de traz pra frente pois o nome dos pesquisadores geralmente está no final do documento
		
		if int(str(paginas[i]).split(":")[1][:-1])>3: # não verifica nas paginas iniciais
			conteudo=paginas[i].extract_text() # extrai o texto da pagina analisada
			
			
			if "equipe técnica".upper() in conteudo :#  se aparecer a frase equipe tecnica, então executa o if
				lista=conteudo.split("\n") # transforma o contudo numa lista
				inicia_equipe=False
				for item in lista: # percorre a lista
					
					if "equipe técnica".upper() in item: # se a lista tiver a frase equipe tecnica
						inicia_equipe=True
					if "apoio" in item.lower() or "são paulo" in item.lower() or "assinado" in item.lower():
						inicia_equipe=False # Se encontrar a palavra apoio para de incluir nomes na lista
					if inicia_equipe:
						nome=item.split("-")[0].split("–")[0].strip() # Tira os caracteres separadores da lista de nomes
						# exclui das diversas denominações
						if "Dr." in nome:
							nome=nome.split("Dr.")[1].strip()
						if "Dra." in nome:
							nome=nome.split("Dra.")[1].strip()
						if "MSc." in nome:
							nome=nome.split("MSc.")[1].strip()
						if "Msc." in nome:
							nome=nome.split("Msc.")[1].strip()
						if "Eng." in nome:
							nome=nome.split("Eng.")[0].strip()
						if "(FIPT)" in nome:
							nome=nome.split("(FIPT)")[0].strip()
						if "(IPT)" in nome:
							nome=nome.split("(IPT)")[0].strip()
						# se encontrar as 
						if not ('equipe' in nome.lower() or 
						'módulo' in nome.lower() or 
						'consultor' in nome.lower() or
						'área' in nome.lower() or
						'laboratório' in nome.lower() or
						'projeto' in nome.lower() or
						'gestor' in nome.lower() or
						'técnico' in nome.lower() or
						'gerente' in nome.lower() or
						'concentração' in nome.lower() or
						'tecnologia' in nome.lower() or
						'unidade' in nome.lower() or
						'negócio' in nome.lower() or
						'avançado' in nome.lower() or
						'pesquisador' in nome.lower()or
						'administrativo' in nome.lower() or
						len(nome)<=6):
							metadados["Equipe"].append(nome)
			
		if not metadados["Equipe"]==[]:# Se prencher o metadados de equipe
			break # para de percorrer as paginas
	return metadados
            
tempo=time.time()
pasta =os.chdir(r"C:\Users\ericaugustin\OneDrive - IPT\Documentos\Prototipos\2025\DTD2024\Extração de dados\relatório exemplo") 
arquivos= os.listdir()
lista=[]
for arquivo in arquivos:
	if arquivo[-3:].upper()=="PDF":
		lista.append(arquivo)
dados={}
lista.sort()
# abre um arquivo json
with open("newData.json", 'w' ,encoding="utf-8") as file:
	# busca na lista de arquivo os metadados
	for n,arquivo in enumerate(lista):
		print(f"Processando arquivo {arquivo} \nArquivo:{n+1} de {len(lista)} ")
		decorrido=time.time()-tempo
		if n>0:
			print(f"tempo decorrido : {decorrido:.2f}s - tempo estimado restante : {decorrido/(n)*(len(lista)-(n)):.2f}s")
		else:
			print(f"tempo decorrido : {decorrido:.2f}s - tempo estimado restante : {(1+random.random()/5)*len(lista):.2f}s")
		dado=busca_metadados(arquivo)
		dados[arquivo]=dado
		print(f"Arquivo {arquivo} finalizado!\n###################")
		
	# grava os metadados no arquivo json
	file.write(json.dumps(dados, sort_keys=True, indent=4,ensure_ascii=False))
#finaliza o script informando po tempo gasto
print("Tempo total gasto para processar todos os arquivos: "+"{0:.2f}".format(time.time()-tempo)+"s")

