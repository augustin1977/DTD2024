from unidecode import unidecode
import math
from datetime import date

def normaliza_texto(texto):
    texto_normalizado = unidecode(texto)
    return texto_normalizado

def calcula_pontos(ano):
    ano_atual=date.today().year
    k=0.1
    result=int(max(1,100*math.exp(k * (ano - ano_atual))))
    return result