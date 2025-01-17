import os
import openpyxl
from funcoes_auxiliares import *
import re
import nltk
from nltk.corpus import stopwords
import json
import pandas as pd



class Importacao:
	def __init__(self,pasta):
		"""Inicializa a classe importação"""
		self.pasta=pasta
		self.arquivos=os.listdir(pasta)
		self.dados=[]
		self.json={}
		self.stop_words=None
		
	def abre_arquivo(self,arquivo):
		if ".XLSX" in arquivo.upper():
			# print(f"Processando o arquivo {arquivo}")
			self.workbook = openpyxl.load_workbook(arquivo)
			self.sheet=self.workbook["Sheet1"]
		else:
			print(f"Arquivo {arquivo} não é um arquivo .XLSX e não será importado")
			self.workbook=None
			self.sheet=None
		
	def importa_linha(self,linha):
		
		registro={}
		if self.sheet:
			registro['Tipo_relatorio']=self.sheet["A"+str(linha)].value
			registro['Numero_documento']=self.sheet["B"+str(linha)].value
			registro['Ano']=self.sheet["C"+str(linha)].value
			registro['Area']=self.sheet["D"+str(linha)].value
			registro['Laboratorio']=self.sheet["E"+str(linha)].value
			registro['CRD']=self.sheet["F"+str(linha)].value
			registro['Titulo']=self.sheet["J"+str(linha)].value.upper()
			registro['N_projeto']=self.sheet["K"+str(linha)].value
			registro['Data_emissão']=self.sheet["O"+str(linha)].value
			registro['Palavras_chave']=self.sheet["P"+str(linha)].value.upper().strip().strip(".").split(';')
			registro['Resumo']=self.sheet["Q"+str(linha)].value
			resumo_limpo= self.sheet["Q"+str(linha)].value.lower()
			resumo_limpo=re.sub(r'[^\w\s]', '', resumo_limpo) 
			resumo_limpo = ' '.join([word for word in resumo_limpo.split() if word not in self.stop_words])
			resumo_limpo = " ".join(self.sheet["P"+str(linha)].value.lower().strip().strip(".").split(';'))+' '+resumo_limpo
			registro['Resumo_limpo']=resumo_limpo
			registro['Equipe']=[]
			equipe=self.sheet["S"+str(linha)].value.upper().replace("EQUIPE:","").replace("AUTOR:","").split(";")
			for pessoa in equipe:
				padrao = r'^(.*?)(?= –|-|/)'
				resultado = re.search(padrao, pessoa)
				if resultado:
					resultado=normaliza_texto(resultado.group(1).strip().replace("  "," ").upper())
				else:
					resultado=normaliza_texto(pessoa.upper().strip())
				registro['Equipe'].append(resultado)
				# ~ print(resultado)
				# ~ time.sleep(0.05)
		return registro
	def numero_linhas(self):
		if self.sheet:
			return self.sheet.max_row
		else:
			return 0
	def importa_dados(self):
		print("#####Importanto dados dos arquivos#####")
		nltk.download('stopwords')
		self.stop_words=set(stopwords.words('portuguese'))
		# print(self.stop_words)
		for arquivo in self.arquivos:
			abre_arquivo=os.path.join(self.pasta,arquivo)
			self.abre_arquivo(abre_arquivo)
			for i in range(2,self.numero_linhas()):
				self.dados.append(self.importa_linha(i))
	def cria_json(self):
		print("##### Criando arquvo 'newData.json' #####")
		nome_arquivo="newData.json"
		for registro in self.dados:
			if not (registro['Numero_documento'] in self.json):
				self.json[registro['Numero_documento']]=registro
			else:
				self.json[registro['Numero_documento']+"A"]=registro
				print(f"Registro {registro['Numero_documento']} repetido, criando o registro {registro['Numero_documento']+'A'}")
		with open(os.path.join(self.pasta,nome_arquivo), 'w' ,encoding="utf-8") as file:
			file.write(json.dumps(self.json, sort_keys=True, indent=4,ensure_ascii=False))
	def recupera_json(self):
		for arquivo in self.arquivos:
			if ".json" in arquivo:
				arquivo_json=os.path.join(self.pasta,arquivo)
				with open(os.path.join(self.pasta,'newData.json'), 'r', encoding='utf-8') as f:
					dados = json.load(f)

					# Converta o JSON em um DataFrame
					df = pd.DataFrame.from_dict(dados, orient='index')
					df.reset_index(inplace=True)
					df.rename(columns={'index': 'Numero_documento'}, inplace=True)
					self.dataframe=df
	def get_df(self):
		print("Recuperando json")
		self.recupera_json()
		return self.dataframe