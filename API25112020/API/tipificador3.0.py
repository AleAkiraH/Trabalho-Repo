from PIL import Image
import imagehash
import os
import datetime
from flask import Flask, jsonify, request
import json
import base64
from io import BytesIO
import cv2
import numpy as np
import imutils
from operator import itemgetter
import face_recognition
import uuid
import imutils
import math 

tm = 0.5

app = Flask(__name__)

def tratarimagem(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	(h, w) = image.shape[:2]
	# if(w > h):
	# 	img = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
	img = image

	thresholdbrilho = 220
	is_light = np.mean(img)
	print(str(is_light))

	if is_light < thresholdbrilho:
		#divisores para clareamento
		divisorgamma = [4.0, 3.7, 3.5,3.2,3.0, 2.8 , 2.5, 2.3 , 2.0 , 1.8 , 1.5 , 1.3 , 1.0]
		for div in divisorgamma:

			invGamma = 1.0 / div
			table = np.array([((i / 255.0) ** invGamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")
			imgteste = cv2.LUT(img, table)
			print(" tentativa " +str(np.mean(imgteste)))
			#faz teste para ver o brilho
			if(np.mean(imgteste) < 220 and np.mean(imgteste) > 200):
				img = imgteste
				break
		
		ret,thresh3 = cv2.threshold(img, 180 ,255,cv2.THRESH_BINARY)
		blur = cv2.GaussianBlur(thresh3,(5,5),0)
		filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
		sharpen_img_1=cv2.filter2D(blur,-1,filter)
		ret,image = cv2.threshold(sharpen_img_1,210,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	else:
		blur = cv2.GaussianBlur(img,(5,5),0)
		filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
		sharpen_img_1=cv2.filter2D(blur,-1,filter)
		tr3, image = cv2.threshold(sharpen_img_1,210,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	return image

# 固定尺寸
def resizeImg(image, height=900):
	h, w = image.shape[:2]
	pro = height / h
	size = (int(w * pro), int(height))
	img = cv2.resize(image, size)
	return img

# 代码承接上文
# 求出面积最大的轮廓
def findMaxContour(image):
	# 寻找边缘
	_, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# 计算面积
	max_area = 0.0
	max_contour = []
	for contour in contours:
		currentArea = cv2.contourArea(contour)
		if currentArea > max_area:
			max_area = currentArea
			max_contour = contour
	return max_contour, max_area

# 代码承接上文
# 多边形拟合凸包的四个顶点
def getBoxPoint(contour):
	# 多边形拟合凸包
	hull = cv2.convexHull(contour)
	epsilon = 0.02 * cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(hull, epsilon, True)
	approx = approx.reshape((len(approx), 2))
	return approx

# 代码承接上文
# 适配原四边形点集
def adaPoint(box, pro):
	box_pro = box
	if pro != 1.0:
		box_pro = box/pro
	box_pro = np.trunc(box_pro)
	return box_pro


# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
def orderPoints(pts):
	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


# 计算长宽
def pointDistance(a, b):
	return int(np.sqrt(np.sum(np.square(a - b))))


# 透视变换
def warpImage(image, box):
	w, h = pointDistance(box[0], box[1]), \
		   pointDistance(box[1], box[2])
	dst_rect = np.array([[0, 0],
						 [w - 1, 0],
						 [w - 1, h - 1],
						 [0, h - 1]], dtype='float32')
	M = cv2.getPerspectiveTransform(box, dst_rect)
	warped = cv2.warpPerspective(image, M, (w, h))
	return warped

def ajustar_docs_em_img(image):
	# image = cv2.imread(path)
	ratio = 900 / image.shape[0]
	img = resizeImg(image)
	binary_img = getCanny(img)
	max_contour, max_area = findMaxContour(binary_img)
	boxes = getBoxPoint(max_contour)
	boxes = adaPoint(boxes, ratio)
	boxes = orderPoints(boxes)
	# 透视变化
	warped = warpImage(image, boxes)
	return warped

def getCanny(image):
	#binariza imagem
	binary = cv2.GaussianBlur(image, (3, 3), 2, 2)
	#detecta arestas
	binary = cv2.Canny(binary, 60, 120, apertureSize=3)
	#aumenta foco arestas
	kernel = np.ones((3, 3), np.uint8)
	binary = cv2.dilate(binary, kernel, iterations=1)
	return binary

def suave_sharpeado_medianblur_glausian(img):
	
	try:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	except:
		pass
	
	#filtro de sharp´ness
	filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

	#filtro normaliza de forma bilateral
	suave = cv2.bilateralFilter(img, 11, 100, 80) 

	#filtro normaliza imagem pela média
	img2 = cv2.medianBlur(suave,9)

	#filtro remove alguns tons de cinza
	ret,th1 = cv2.threshold(img2,50,200,cv2.THRESH_TOZERO)
	
	#filtro Gausiano para curvas
	th3 = cv2.adaptiveThreshold(th1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	
	return th3

def padronizacao_docs(imgorigem, imgtemplate,nivelthrs = 0.37):
	encontrado = False
	
	if( imgorigem.shape[1] > imgorigem.shape[0]):
		widith = 1280
		height = 720
	else:
		widith = 720
		height = 1280

	imgorigem = cv2.resize(imgorigem,(widith,height))
 
	# #load da imagem pra tentar achar
	# img_gray = cv2.cvtColor(imgorigem, cv2.COLOR_BGR2GRAY)

	#seta o tamanho para pesquisa
	w, h = imgtemplate.shape[::-1]
	#busca na imagem
	res = cv2.matchTemplate(imgorigem,imgtemplate,cv2.TM_CCOEFF_NORMED)
	#threshoul de acerto
	threshold = nivelthrs
	loc = np.where( res >= threshold)
	
	for pt in zip(*loc[::-1]):
		encontrado = True
		cv2.rectangle(imgorigem, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	
	return encontrado

class retorno_final:
	def __init__(self):
		self.lst_retorno_final = []

class listagem_de_resultados:    
	def __init__(self):
		self.documento = ''
		self.score = 0
		self.imagem_base64 = ''

class post_entrada_tipificar:
	def __init__(self, j):
		self.__dict__ = json.loads(j)

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

def getfiles(dir):
	all_files = list()
	for root, dirs, files in os.walk(dir):
		for f in files:
			fullpath = os.path.join(root, f)
			# if os.path.splitext(fullpath)[1] == '.'+ str(extensao) +'':
			all_files.append(str(fullpath))
	return all_files

def gerador_hash_tipificados():
	Documentos = getfiles('Documentos')

	try:
		os.remove('hashs')
	except:
		pass
 	
	tipodoc = ""

	for doc in Documentos:        
		try:
			thresh4 = base64_to_img_npar(file_to_base64(doc))
			if ('RG FRENTE' in doc):
				face_locations = face_recognition.face_locations(thresh4)
				
				# try:
				# except:
				# 	return False, lst_res, ''    
				x1 = (face_locations[0][3] - int((0 + ((thresh4.shape[1]) * 0.12))))
				y1 = int((0 + ((thresh4.shape[0]) * 0.05)))
				x2 = (face_locations[0][1] + int((0 + ((thresh4.shape[1]) * 0.12))))
				y2 = int((thresh4.shape[0] - ((thresh4.shape[0]) * 0.05)))

				cv2.rectangle(thresh4, (x1, y1), (x2, y2), (0,0,0), -1)
				thresh4 = cv2.cvtColor(thresh4, cv2.COLOR_BGR2GRAY)
				thresh4 = cv2.GaussianBlur(thresh4,(5,5),0) 
				ret,thresh4 = cv2.threshold(thresh4,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
				# cv2.imwrite('facefrente'+str(uuid.uuid4())+'.jpg', thresh4)
			if ('RG VERSO' in doc):
				img = cv2.cvtColor(thresh4, cv2.COLOR_BGR2GRAY)
				img = cv2.GaussianBlur(img,(5,5),0) 
				ret,thresh4 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			else:
				ret,thresh4 = cv2.threshold(thresh4,127,255,cv2.THRESH_TOZERO)

			doc_base64 = base64_to_img_pil(img_npar_to_base64(thresh4))
			hash1 = imagehash.average_hash(doc_base64) 
			
			tipodoc = doc.split('\\')[1]
			lsthashs = open('hashs','a')
			lsthashs.write(tipodoc+";"+str(hash1)+"\n")
			lsthashs.close()	
		except:
			print("ERRO:")
			print(doc)

def rotacionar(image, angle):
	image = imutils.rotate_bound(image, angle)
	return image

def percentage(part, whole):
	return 100 * float(part)/float(whole)

def alinhar_imagem(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(image)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	coords = np.column_stack(np.where(thresh > 0))
	angle = cv2.minAreaRect(coords)[-1]
	if angle < -45:
		angle = -(90 + angle)
	else:
		angle = -angle
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	return rotated

def findMaxContour(image):
	contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	max_area = 0.0
	max_contour = []
	all_contour = {}
	for contour in contours:
		currentArea = cv2.contourArea(contour)
		#print(currentArea) # debug
		all_contour.update({currentArea:contour}) 
		if currentArea > max_area:
			max_area = currentArea
			max_contour = contour
	return max_contour, max_area, all_contour

def findMaxContour_alinhamento(image):
	cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	return cnts

def leitor_hash_tipificados():
	list_hashs_tipificados = open('hashs')
	return list_hashs_tipificados

def tipificar_imagem(base64imagem, lst_res, cutoff = 5):
	
	hash_imagem_para_comparar = imagehash.average_hash(base64_to_img_pil(base64imagem))
	
	resultado = []	

	# BUSCAR NO ARQUIVO OS HASHS DAS IMAGENS JA TIPIFICADAS
	list_hashs_tipificados = leitor_hash_tipificados()

	#  DETERMINA O TIPO, CONVERTE O HEX E GERA UM RESULTADO DAS TIPIFICAÇÕES ENCONTRADAS
	for lht in list_hashs_tipificados:
		tipo_documento = lht.split(';')[0] 
		hash_documento = lht.split(';')[1].replace('\n','')
		hash_documento = imagehash.hex_to_hash(hash_documento)	

		if hash_imagem_para_comparar - hash_documento < cutoff:
			resultado.append(tipo_documento)
	
	# AJUSTA OS TIPOS ENCOTNRADOS E ORGANIZA PARA GERAR UM SCORE
	agrupamento_encontrados = set(map(lambda x:x, resultado))
	segregacao_por_linha_dos_encontrados = [[y for y in resultado if y==x] for x in agrupamento_encontrados]

	# REALIZA A SOMA DOS ENCONTRADOS PARA GERAR UM SCORE
	soma_encontrados = 0
	for splde in segregacao_por_linha_dos_encontrados:
		soma_encontrados = (soma_encontrados + len(splde))

	# ADICIONA EM UMA LISTA OS RESULTADO E OS SCORES DA IMAGEM TIPIFICADA
	
	for splde in segregacao_por_linha_dos_encontrados:
		lst_res.score = ("%.2f" % round((len(splde) / soma_encontrados) * 100,2))
		lst_res.documento = splde[0]
		
	# fim1 = datetime.datetime.now()
	# print("Comparação com as imagens HASH geradas, tempo executado:")
	# print(fim1 - inicio1)
 
	return lst_res

def alinhamento(image):
	(height, width) = image.shape[:2]
	retanguloslist = []
	binary_img = getCanny(image)
	img = image.copy()
	img2 = image.copy()
	#cv2.imshow('canny', cv2.resize(binary_img,(1000,1000)))
	all_contour = findMaxContour_alinhamento(binary_img)
	retanguloencontrado = False
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	for c in all_contour:
		x,y,w,h = cv2.boundingRect(c)
		if (w > (width * 0.80) and h > ( height * 0.80)  ):
			retanguloencontrado = True
			# cv2.rectangle(img2, (x, y), (x + w, y + h), (36,255,12), 2)
			rect = cv2.minAreaRect(c)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			# cv2.drawContours(img2,[box],0,(0,0,255),1)
			
			box = sorted(box ,  key=lambda x: (x[1], -x[1] ), reverse=True )  

			catetoo = box[0][1] - box[1][1]   
			
			if(box[0][0] > box[1][0]):
				catetoa = box[0][0] - box[1][0]
			else:
				catetoa = box[1][0] - box[0][0]
			
			hipotenusa = math.hypot(catetoa,catetoo)
			seno = catetoo/hipotenusa
			graus = np.degrees(math.asin(seno))
			# img = cv2.circle(img2, (box[0][0],box[0][1]), 2, (0,0,255), 2)
			# img = cv2.circle(img2, (box[1][0],box[1][1]), 5, (0,0,255), 2)
			# img = cv2.circle(img2, (box[2][0],box[2][1]), 10, (0,0,255), 2)
			# img = cv2.circle(img2, (box[3][0],box[3][1]), 20, (0,0,255), 2)
		
			if(box[1][0] > box[0][0]):
				
				graus = graus * -1

			rot_mat = cv2.getRotationMatrix2D(image_center, graus, 1.0)
			image = cv2.warpAffine(img2, rot_mat, image.shape[1::-1], borderValue=(255,255,255))
			#r = imutils.rotate(img2, graus) 
			cv2.imwrite('alinhamento.jpg',image) 
			# cv2.imshow('com alinhamento',r)

	return image
	# cv2.imshow('sem alinhamento',img2)
	# cv2.waitKey(0)

@app.route("/gerarhash", methods=['GET'])
def gerarhash():
	gerador_hash_tipificados()

	return "Gerado com sucesso"

def tip_docs_rg_verso(crop_img, rot, cutoff, lst_res):
	# VALIDAR SE É UM RG VERSO
	if (True):
		try:
			crop_img_rotacionada = rotacionar(crop_img, 90 * rot)
		except:
			return False, lst_res, img1
		# cv2.imshow('rot_crop_img', cv2.resize(crop_img_rotacionada,None,fx=tm,fy=tm))
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		# ret,thresh4 = cv2.threshold(crop_img_rotacionada,127,255,cv2.THRESH_TOZERO)
		img1 = img_npar_to_base64(crop_img_rotacionada)
		lst_res = tipificar_imagem(img1,lst_res, cutoff)
	  
		# print(retorno)
		return True, lst_res, img1

def tip_docs_rg_frente(crop_img, rot, cutoff, lst_res):
	# VALIDAR SE É UM RG FRENTE
	if (True):
		try:
			crop_img_rotacionada = rotacionar(crop_img, 90 * rot)
		except:
			return False, lst_res, ''
		
		face_locations = face_recognition.face_locations(crop_img_rotacionada)
		
		ret,thresh4 = cv2.threshold(crop_img_rotacionada,127,255,cv2.THRESH_TOZERO)    
		
		try:
			x1 = (face_locations[0][3] - int((0 + ((thresh4.shape[1]) * 0.12))))
		except:
			return False, lst_res, ''    
		y1 = int((0 + ((thresh4.shape[0]) * 0.05)))
		x2 = (face_locations[0][1] + int((0 + ((thresh4.shape[1]) * 0.12))))
		y2 = int((thresh4.shape[0] - ((thresh4.shape[0]) * 0.05)))
		
		percx = (face_locations[0][1] / thresh4.shape[1])*100
		percy = (face_locations[0][0] / thresh4.shape[0])*100

		cv2.rectangle(thresh4, (x1, y1), (x2, y2), (0,0,0), -1)
		thresh4 = cv2.cvtColor(thresh4, cv2.COLOR_BGR2GRAY)
		thresh4 = cv2.GaussianBlur(thresh4,(5,5),0) 
		ret3,thresh4 = cv2.threshold(thresh4,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		
		img1 = img_npar_to_base64(thresh4)
		
		if (percx > 50):
			lst_res = tipificar_imagem(img1,lst_res, cutoff)
		else:		
			lst_res.score = 0
			lst_res.documento = ''
		# print(retorno)
		return True, lst_res, img1

def tip_docs_brancos(img, cutoff, lst_res, rotacionar_documento = False):    
	ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
	
	if (rotacionar_documento):
		for rot in range(0,4):
			rotacionada = rotacionar(thresh4, 90 * rot)
			# cv2.imshow('img' + str(rot), cv2.resize(rotacionada,None,fx=tm,fy=tm))
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			img1 = img_npar_to_base64(rotacionada)
			lst_res = tipificar_imagem(img1,lst_res,cutoff)
			if (len(lst_res.documento) > 0):
				return True, lst_res, img1
	else:
		img1 = img_npar_to_base64(thresh4)
		lst_res = tipificar_imagem(img1,lst_res,cutoff)
		if (len(lst_res.documento) > 0):
			return True, lst_res, img1
	return False, lst_res, img1

@app.route("/rg", methods=['POST'])
def rg():
	tipificar_post = post_entrada_tipificar(json.dumps(request.get_json()))
	img = base64_to_img_npar(tipificar_post.img1)
	
	face_locations = face_recognition.face_locations(img)
	
	ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)    
	
	x1 = (face_locations[0][3]-80)
	y1 = int((0 + ((thresh4.shape[0]) * 0.05)))
	x2 = (face_locations[0][1]+95)
	y2 = int((thresh4.shape[0] - ((thresh4.shape[0]) * 0.05)))

	cv2.rectangle(thresh4, (x1, y1), (x2, y2), (0,0,0), -1)
	
	cv2.imshow('', thresh4)
	cv2.waitKey(0)
	img1 = img_npar_to_base64(thresh4)
	
	retorno = tipificar_imagem(img1, tipificar_post.cutoff)   
	return retorno

@app.route("/tipificar", methods=['POST'])
def tipificar():
	
	try:
		os.remove('crop_img_facematch1.jpg')
	except:
		pass
	try:
		os.remove('crop_img_facematch2.jpg')
	except:
		pass
	try:
		os.remove('crop_img_facematch3.jpg')
	except:
		pass
	try:
		os.remove('crop_img_facematch4.jpg')
	except:
		pass
	try:
		os.remove('crop_img_facematch5.jpg')
	except:
		pass
	try:
		os.remove('docbranco.jpg')
	except:
		pass
	try:
		os.remove('alinhamento.jpg')
	except:
		pass
	try:
		os.remove('Original.jpg')
	except:
		pass
	try:
		os.remove('RG VERSO0.jpg')
	except:
		pass
	try:
		os.remove('RG VERSO1.jpg')
	except:
		pass
	try:
		os.remove('RG VERSO2.jpg')
	except:
		pass
	try:
		os.remove('RG VERSO3.jpg')
	except:
		pass
	try:
		os.remove('RG VERSO4.jpg')
	except:
		pass
	try:
		os.remove('RG VERSO00.jpg')
	except:
		pass
	try:
		os.remove('RG .jpg')
	except:
		pass
	try:
		os.remove('recorte.jpg')
	except:
		pass
	try:
		os.remove('recorte1.jpg')
	except:
		pass
	try:
		os.remove('recorte2.jpg')
	except:
		pass
	try:
		os.remove('recorte3.jpg')
	except:
		pass
	try:
		os.remove('recorte4.jpg')
	except:
		pass
	try:
		os.remove('recorte5.jpg')
	except:
		pass
	try:
		os.remove('recorte6.jpg')
	except:
		pass
	try:
		os.remove('recorte7.jpg')
	except:
		pass
	try:
		os.remove('recorte8.jpg')
	except:
		pass
	try:
		os.remove('recorte9.jpg')
	except:
		pass
	try:
		os.remove('recorte10.jpg')
	except:
		pass
	try:
		os.remove('RG FRENTE1.jpg')
	except:
		pass
	try:
		os.remove('RG FRENTE2.jpg')
	except:
		pass
	try:
		os.remove('RG FRENTE3.jpg')
	except:
		pass
	try:
		os.remove('RG FRENTE4.jpg')
	except:
		pass
	try:
		os.remove('RG FRENTE0.jpg')
	except:
		pass
	rtn = retorno_final()

	tipificar_post = post_entrada_tipificar(json.dumps(request.get_json()))
	img = base64_to_img_npar(tipificar_post.img1)
	cv2.imwrite('Original.jpg',img)
	# cv2.imshow('Original',cv2.resize(img,None,fx=tm,fy=tm))
	# cv2.waitKey(0)
	# TIPIFICAR DOCUMENTOS BRANCOS
	lst_res = listagem_de_resultados()
	boolvar, lst_res, imgBase64 = tip_docs_brancos(img, tipificar_post.cutoff, lst_res)
	if (boolvar):
		if not ("RG " in lst_res.documento):
			lst_res.imagem_base64 = imgBase64
			cv2.imwrite('docbranco.jpg',base64_to_img_npar(imgBase64))
			# cv2.imshow('retorno doc branco', cv2.resize(base64_to_img_npar(imgBase64),None,fx=tm,fy=tm))
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			rtn.lst_retorno_final.append(lst_res)
			return json.dumps(rtn, default=lambda o: o.__dict__)

	# BINARIZAR IMAGEM E TRACAR CONTORNOS
	# img2 = tratarimagem(img)
	binary_img = getCanny(img)
	max_contour, max_area, all_contour = findMaxContour(binary_img)
	
	# Desenha os 4 maiores contornos encontrados
	dict_sorted = sorted(all_contour.items(), key=itemgetter(0), reverse=True)
	if (len(dict_sorted) > 50):
			rtn = retorno_final()
			rtn.lst_retorno_final.append(lst_res)
			return json.dumps(rtn, default=lambda o: o.__dict__)
	
	top4 = dict_sorted[:5] # top boxes a serem selecionados

	num_crop = 0
	for box in top4:
		if box[0] > 5000: # teste threshold por area de contorno dimensoes de entrada 

			num_crop  = num_crop + 1
			contr = box[1]
			x,y,w,h = cv2.boundingRect(contr)
			# cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 3)
			
			#Faz crop da box encontrada
			crop_img = img[y:y+h, x:x+w]
			
			# # # (h, w) = crop_img.shape[:2]
			# # # newimage = np.zeros((h + int(h*0.1),w + int(w*0.1), 3), np.uint8)
			# # # newimage[:] = 255
			# # # centery = int( (newimage.shape[0]/2) - (h/2) )
			# # # centerx = int ( (newimage.shape[1]/2) - (w/2) )
			# # # newimage[ centery : centery + h ,centerx : centerx + w ] = crop_img
			# # # # cv2.imshow('img', newimage )
			# # # # cv2.waitKey(0)
			# # # crop_img = newimage
			crop_img_facematch = img[y:y+h, x:x+w]
			cv2.imwrite('crop_img_facematch'+str(num_crop)+'.jpg',crop_img_facematch)
			# crop_img = suave_sharpeado_medianblur_glausian(crop_img)
			crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
			crop_img = cv2.GaussianBlur(crop_img,(5,5),0) 
			# ret3,crop_img = cv2.threshold(crop_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			# crop_img = getCanny(crop_img)
			cv2.imwrite('recorte'+str(num_crop)+'.jpg',crop_img)
			crop_img = alinhamento(crop_img)
			
			crop_lst_asp_rat = []
			
			if (crop_img.shape[0] > crop_img.shape[1]):
				asp_rat = crop_img.shape[0] / crop_img.shape[1]
				if (asp_rat > 2.5) and (asp_rat < 6):
					crop_x = crop_img.shape[1]
					crop_y = crop_img.shape[0]
					crop_asp_rat = crop_img[0:int(crop_y/2), 0:crop_x]
					crop_lst_asp_rat.append(crop_asp_rat)
					crop_asp_rat = crop_img[int(crop_y/2):crop_y, 0:crop_x]
					crop_lst_asp_rat.append(crop_asp_rat)
			else:
				asp_rat = crop_img.shape[1] / crop_img.shape[0]
				if (asp_rat > 2.5) and (asp_rat < 6):
					crop_x = crop_img.shape[1]
					crop_y = crop_img.shape[0]
					crop_asp_rat = crop_img[0:crop_y,0:int(crop_x/2)]
					crop_lst_asp_rat.append(crop_asp_rat)
					crop_asp_rat = crop_img[0:crop_y,int(crop_x/2):crop_x]
					crop_lst_asp_rat.append(crop_asp_rat)
			
			if (len(crop_lst_asp_rat)>0):
				for c in crop_lst_asp_rat:
					ret3,crop_img = cv2.threshold(c,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
					# cv2.imshow('img',crop_img)
					# cv2.waitKey(0)
					for rot in range(0,4):
					
						lst_res = listagem_de_resultados()

						boolvar, lst_res, imgBase64 = tip_docs_rg_verso(crop_img, rot, 6, lst_res)
						# if not (len(lst_res.documento) > 0):
						# 	continue
						if (boolvar):
							if ("RG VERSO" in lst_res.documento):
								lst_res.imagem_base64 = imgBase64
								cv2.imwrite('RG VERSO'+str(rot)+'.jpg',crop_img)
								# cv2.imshow('retorno RG VERSO', cv2.resize(base64_to_img_npar(imgBase64),None,fx=tm,fy=tm))
								# cv2.waitKey(0)
								# cv2.destroyAllWindows()
								rtn.lst_retorno_final.append(lst_res)
								continue
						lst_res = listagem_de_resultados()
						boolvar, lst_res, imgBase64 = tip_docs_rg_frente(crop_img_facematch, rot, tipificar_post.cutoff, lst_res)
						if (boolvar):
							if ("RG FRENTE" in lst_res.documento):
								lst_res.imagem_base64 = imgBase64
								cv2.imwrite('RG FRENTE'+str(rot)+'.jpg',base64_to_img_npar(imgBase64))
								# cv2.imshow('RG FRENTE', cv2.resize(base64_to_img_npar(imgBase64),None,fx=tm,fy=tm))
								# cv2.waitKey(0)
								# cv2.destroyAllWindows()
								rtn.lst_retorno_final.append(lst_res)
						else:
							continue
			else:
				for rot in range(0,4):
					
					lst_res = listagem_de_resultados()

					boolvar, lst_res, imgBase64 = tip_docs_rg_verso(crop_img, rot, 6, lst_res)
					# if not (len(lst_res.documento) > 0):
					# 	continue
					if (boolvar):
						if ("RG VERSO" in lst_res.documento):
							lst_res.imagem_base64 = imgBase64
							cv2.imwrite('RG VERSO'+str(rot)+'.jpg',crop_img)
							# cv2.imshow('retorno RG VERSO', cv2.resize(base64_to_img_npar(imgBase64),None,fx=tm,fy=tm))
							# cv2.waitKey(0)
							# cv2.destroyAllWindows()
							rtn.lst_retorno_final.append(lst_res)
							continue
					lst_res = listagem_de_resultados()
					boolvar, lst_res, imgBase64 = tip_docs_rg_frente(crop_img_facematch, rot, tipificar_post.cutoff, lst_res)
					if (boolvar):
						if ("RG FRENTE" in lst_res.documento):
							lst_res.imagem_base64 = imgBase64
							cv2.imwrite('RG FRENTE'+str(rot)+'.jpg',base64_to_img_npar(imgBase64))
							# cv2.imshow('RG FRENTE', cv2.resize(base64_to_img_npar(imgBase64),None,fx=tm,fy=tm))
							# cv2.waitKey(0)
							# cv2.destroyAllWindows()
							rtn.lst_retorno_final.append(lst_res)
					else:
						continue
	# return str(rtn.lst_retorno_final)
	return json.dumps(rtn, default=lambda o: o.__dict__)
			
@app.route("/", methods=['GET'])
def getsimples():
	return "Api Tipificação - {}".format(datetime.datetime.now())

if __name__ == '__main__':
	print ('API_TIPIFICAÇÃO DE DOCUMENTOS INICIADA')
	app.run(host='127.0.0.1', port=3000)
 
