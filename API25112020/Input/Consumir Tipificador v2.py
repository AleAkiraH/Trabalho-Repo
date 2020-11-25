from requests import post
from json import dumps, loads
from os import listdir
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

def file_to_base64(caminho_imagem):
    b64 = base64.b64encode(open(caminho_imagem,"rb").read())    
    return b64.decode("utf8")

def base64_to_img_pil(base64_string):
    if(isinstance(base64_string, str)):
        imagem = Image.open(BytesIO(base64.b64decode(base64_string)))
    return imagem

def base64_to_img_npar(base64_string):
    buffer = cv2.imencode('.jpg', np.asarray(base64_to_img_pil(base64_string)))[1]
    image_bs64 = base64.b64encode(buffer) 
    return np.array(Image.open(BytesIO(base64.b64decode(image_bs64))))

def img_npar_to_base64(np_imagem):
    output_bytes = BytesIO()
    pil_img = Image.fromarray(np_imagem)
    pil_img.save(output_bytes, 'JPEG')
    bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')    
    return base64_str

def chamartipificação(Base64,Threshold):
    
    url = 'http://localhost:3000/tipificar'
    # url = 'http://c30f53a8b97f.ngrok.io/tipificar'
    
    headers = {'Content-Type': 'application/json'}
    parametros = {'img1': Base64, 'cutoff': Threshold}

    #return loads(post(url, data=dumps(parametros), headers=headers).content)
    return str(post(url, data=dumps(parametros), headers=headers).content)

inicio  = datetime.now()

print(f"Inicio: {inicio}")

diretorio = 'RG'

listaarquivos = listdir(diretorio)

total = listaarquivos.__len__()/2

for nomearquivo in listaarquivos:
    
    base64_img = file_to_base64(diretorio + '\\' + nomearquivo)

    #Consome o tipificador
    retorno = chamartipificação(base64_img,5).replace("b'",'').replace("}'",'}')
    import json
    retorno = json.loads(retorno)
    # # cv2.waitKey(0)
    if not len(retorno['lst_retorno_final']) == 0:
        for it in retorno['lst_retorno_final']:
            # abre o arquivo de log
            log = open('Saida Teste.txt', 'a', encoding='utf-8')
            #escrever retorno no log
            log.write(nomearquivo.replace('base64.txt','') + ';' + str(it['documento']) + ';6;' + str(it['score']) +'\n')
            #Fecha o arquivo de log
            log.close()
    else:
          # abre o arquivo de log
            log = open('Saida Teste.txt', 'a', encoding='utf-8')
            #escrever retorno no log
            log.write(nomearquivo.replace('base64.txt','') + ';;6;0\n')
            #Fecha o arquivo de log
            log.close()

    #     documento = '{ }'
    
    # else:
    #     pass
    #     # # #Busca o maior score retornado
    #     # # retornolista = retorno.replace(',"documento": ','"documento":').replace('{ "documento": ','').replace(' }','').split('"documento":')
    #     retornolista = ''
    #     # # documento = ""
    #     # # maior = 0.0
    #     # # for item in retornolista:
    #     # #     x = item.replace('"','').replace('%','').replace(' ','').split(':')
    #     # #     score = float(x[1])
    #     # #     if score > maior:
    #     # #         maior = score
    #     # #         documento = item 

    
    print(nomearquivo)
    print(f'{total} - Threshold {5}')
    total -= 1 


fim = datetime.now()

print(f"Inicio: {inicio}")
print('-'*30)
print(f"Fim: {fim}")
print('-'*30)
print(f"Duração: {fim-inicio}")

