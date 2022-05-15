import heapq
import re
import string

import nltk
import numpy
import spacy
from goose3 import Goose

pln = spacy.load("pt_core_news_sm")

nltk.download("stopwords")
nltk.download("stopwords")

g = Goose()
url = "https://iaexpert.academy/2020/11/09/ia-preve-resultado-das-eleicoes-americanas/"
artigo_internet = g.extract(url)

texto_original = artigo_internet.cleaned_text

def limpa_texto(txt):
  txt = re.sub (r'\s', ' ', txt)
  return txt

def format_lemma(txt):
  txt = limpa_texto(txt)
  txt = txt.lower()

  documento = pln(txt)
  tokens = []
  stopwords = nltk.corpus.stopwords.words("portuguese")

  for token in documento:
    tokens.append(token.lemma_)
  
  tokens = [palavra for palavra in tokens if palavra not in stopwords and palavra not in string.punctuation]
  texto_formatado = " ".join(element for element in tokens if not element.isdigit())

  return texto_formatado



def calcula_scor_sentenca(sent_lista, top_palavras, distancia):
  notas = []
  indice_sentenca = 0
  for sentenca in [nltk.word_tokenize(sentenca) for sentenca in sent_lista]:
    index_palavra = []
    for palavra in top_palavras:
      try:
        index_palavra.append(sentenca.index(palavra))
      except ValueError:
        pass
    index_palavra.sort()

    if len(index_palavra) == 0:
      continue

    lista_grupos = []
    grupo = [index_palavra[0]]

    cont = 1
    while cont < len(index_palavra):
      if (index_palavra[cont] - index_palavra[cont-1]) < distancia:
        grupo.append(index_palavra[cont])

      else:
        lista_grupos.append(grupo[:])
        grupo = [index_palavra[cont]]
      cont += 1
      lista_grupos.append(grupo)


      nota_maxima_grupo = 0
      for grupo in lista_grupos:
        quant_palavras_importantes = len(grupo)
        total_palavras_grupo = grupo[-1] + grupo[0] + 1

        nota = 1.0 * quant_palavras_importantes**2/ total_palavras_grupo

        if nota > nota_maxima_grupo:
          nota_maxima_grupo = nota

    notas.append((nota_maxima_grupo, indice_sentenca))
    indice_sentenca += 1
  return notas

def sumariza_lemma(txt, n_palavras, distancia, quant_sentencas):
    sentencas_original = [limpa_texto(sent) for sent in nltk.sent_tokenize(txt)]
    sentencas_format = [format_lemma(sent) for sent in sentencas_original]

    palavras = [palavra for sent in sentencas_format for palavra in nltk.word_tokenize(sent)]

    freq = nltk.FreqDist(palavras)
    top_palavras = [palavra[0] for palavra in freq.most_common(n_palavras)]

    nota_sentencas = calcula_scor_sentenca(sentencas_format, top_palavras, distancia)

    melhores_sentencas = heapq.nlargest(quant_sentencas, nota_sentencas)

    melhores_sentencas = [sentencas_original[i] for (nota, i) in melhores_sentencas]
    return sentencas_original, melhores_sentencas, nota_sentencas

