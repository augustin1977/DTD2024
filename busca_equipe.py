from importando_dados import *
import os

root=os.getcwd()
endereco=os.path.join(root,"Extração de dados","Relatórios Biblioteca")
print(endereco)
planilha=Importacao(endereco)
planilha.importa_dados()
planilha.cria_json()
equipe=[]
for registro in planilha.json:
    for pessoa in planilha.json[registro]["Equipe"]:
        if pessoa not in equipe:
            equipe.append(pessoa)
equipe.sort()
for pessoa in equipe:
    print(pessoa)