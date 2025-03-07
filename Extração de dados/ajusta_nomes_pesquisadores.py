import os
import openpyxl
caminho=os.path.join(os.getcwd(),'Extração de dados','pesquisadores_ativos','dicionario_nomes.csv')
print(caminho)
arquivo=open(caminho,'r',encoding='utf-8')
dados=arquivo.readlines()
arquivo.close()
dicionario={}
for linha in dados:
    linha=linha.strip().split(';')
    dicionario[linha[0]]=linha[1]
    

pasta=os.path.join(os.getcwd(),'Extração de dados','Relatórios Biblioteca')
arquivos=os.listdir(pasta)
for arquivo in arquivos:
    if ".XLSX" in arquivo.upper():
        arquivo_excel=os.path.join(pasta,arquivo)   
        print(f"Processando o arquivo {arquivo}")
        workbook = openpyxl.load_workbook(arquivo_excel)
        sheet=workbook["Sheet1"]
    else:
        print(f"Arquivo {arquivo} não é um arquivo .XLSX e não será importado")
        sheet=None
        
    if sheet:
        for linha in range(2,sheet.max_row+1):
            sheet[f'S{linha}'].value=sheet[f'S{linha}'].value.upper()+";"
            for nome in dicionario:
                nome_compara1=nome.upper()+" ;"
                nome_compara2=nome.upper()+";"
                nome_compara3=nome.upper()+"\n"
                
                
                if nome == sheet[f'R{linha}'].value.upper():
                    valor=sheet[f'R{linha}'].value.upper()
                    valor_novo=valor.replace(nome,dicionario[nome])
                    sheet[f'R{linha}'].value =valor_novo 
                    print(f"Nome {nome} encontrado na linha {linha} da coluna R e substituido por {valor_novo}")                
 
                if nome_compara1 in sheet[f'S{linha}'].value.upper():
                    valor=sheet[f'S{linha}'].value.upper()
                    valor_novo=valor.replace(nome_compara1,dicionario[nome]+";")
                    sheet[f'S{linha}'].value =valor_novo
                    print(f"Nome {nome} encontrado na linha {linha} da coluna S e substituido por {dicionario[nome]}")
                elif nome_compara2 in sheet[f'S{linha}'].value.upper():
                    valor=sheet[f'S{linha}'].value.upper()
                    valor_novo=valor.replace(nome_compara2,dicionario[nome]+";")
                    sheet[f'S{linha}'].value =valor_novo
                    print(f"Nome {nome} encontrado na linha {linha} da coluna S e substituido por {dicionario[nome]}")
                elif nome_compara3 in sheet[f'S{linha}'].value.upper():
                    valor=sheet[f'S{linha}'].value.upper()
                    valor_novo=valor.replace(nome_compara3,dicionario[nome]+";")
                    sheet[f'S{linha}'].value =valor_novo
                    print(f"Nome {nome} encontrado na linha {linha} da coluna S e substituido por {dicionario[nome]}")
            sheet[f'S{linha}'].value=sheet[f'S{linha}'].value.strip(";")
        workbook.save(arquivo_excel)
        workbook.close()
        print(f"\n########################################\nArquivo {arquivo} ajustado com sucesso\n########################################\n")
                
print("Fim do processamento")    
