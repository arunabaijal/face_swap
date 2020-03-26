# from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import random

def findTuple(pts,pt):
	p1 = pt[0]
	p2 = pt[1]

	for ind, point in enumerate(pts):
		if(point[0]==p1):
			if(point[1]==p2):
				return ind
	print('element not found!!')
	return 'not found'

def getTriIndices(subdiv,points,img):
	triangles = subdiv.getTriangleList()
	indexes_triangles = []
	for t in triangles:
		pt1 = (int(t[0]), int(t[1]))
		pt2 = (int(t[2]), int(t[3]))
		pt3 = (int(t[4]), int(t[5]))

		index_pt1 = findTuple(points,pt1)
		# index_pt1 = np.where((points == pt1))
		# print(index_pt1)
		# print(pt1)
		# index_pt1 = extract_index_nparray(index_pt1)

		index_pt2 = findTuple(points,pt2)
		# index_pt2 = np.where((points == pt2))
		# print(index_pt2)
		# print(pt2)
		# index_pt2 = extract_index_nparray(index_pt2)

		index_pt3 = findTuple(points,pt3)
		# index_pt3 = np.where((points == pt3))
		# print(index_pt3)
		# print(pt3)
		# index_pt3 = extract_index_nparray(index_pt3)

		if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
			triangle = [index_pt1, index_pt2, index_pt3]
			indexes_triangles.append(triangle)
		# cv2.circle(img, pt1, 1, (0, 0, 255), 4)
		# cv2.circle(img, pt2, 1, (0, 0, 255), 4)
		# cv2.circle(img, pt3, 1, (0, 0, 255), 4)
		cv2.line(img, pt1, pt2, (0, 0, 255), 2)
		cv2.line(img, pt2, pt3, (0, 0, 255), 2)
		cv2.line(img, pt1, pt3, (0, 0, 255), 2)

		# cv2.imshow('denauly1',img)
		# cv2.waitKey(0)

	return indexes_triangles

def extract_index_nparray(nparray):
	index = None
	for num in nparray[0]:
		index = num
		break
	return index

def calBarycentricInv(subdiv):
	triangleList = subdiv.getTriangleList()
	bList = []
	for t in triangleList :
		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])
		B = np.array([[pt1[0],pt2[0],pt3[0]],[pt1[1],pt2[1],pt3[1]],[1,1,1]])
		# print(np.linalg.inv(B))
		bList.append(np.linalg.inv(B))
	return bList

def calBarycentric(subdiv):
	triangleList = subdiv.getTriangleList()
	bList = []
	for t in triangleList :
		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])
		B = np.array([[pt1[0],pt2[0],pt3[0]],[pt1[1],pt2[1],pt3[1]],[1,1,1]])
		# print(np.linalg.inv(B))
		bList.append(np.linalg.inv(B))
	return bList

# def checkInside(xy,subdiv,bList):


# Check if a point is inside a rectangle
def rect_contains(rect, point) :
	if point[0] < rect[0] :
		return False
	elif point[1] < rect[1] :
		return False
	elif point[0] > rect[2] :
		return False
	elif point[1] > rect[3] :
		return False
	return True

# Draw a point
def draw_point(img, p, color ) :
	cv2.circle( img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0 )

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :

	triangleList = subdiv.getTriangleList();
	size = img.shape
	r = (0, 0, size[1], size[0])

	for t in triangleList :
		
		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])
		
		if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
		
			cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
			cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
			cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

def getTri(image):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	img_orig = image.copy();

	# image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	rect = rects[0]
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = shape_to_np(shape)
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = rect_to_bb(rect)
	# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# # show the face number
	# cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
	# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	points = []
	for (x, y) in shape:
		points.append((int(x), int(y)))
		cv2.circle(image, (x, y), 1, (0, 255, 0), 4)

	#expanding the rect to fit the feature points
	rect_exp = 20
	rect_x = sorted([0,rect.left()-rect_exp])[1]
	rect_y = sorted([0,rect.top()-rect_exp])[1]
	rect_h = rect.width()+2*rect_exp
	rect_w = rect.height()+2*rect_exp

	# Dividing only for the face rect into triangles
	subdiv = cv2.Subdiv2D((rect_x, rect_y,rect_h, rect_w));
	# Dividing entire image into triangles
	# subdiv = cv2.Subdiv2D((0, 0,image.shape[1], image.shape[0]));
	for p in points:
		subdiv.insert(p)
		# Show animation
	# bList = calBarycentric(subdiv)
	# draw_delaunay(image, subdiv, (255, 255, 255) )
	# cv2.imshow('voronoi',image)
	# cv2.waitKey(0)


	return subdiv,image,points

# Draw voronoi diagram
def draw_voronoi(img, subdiv) :
	# ( facets, centers) = subdiv.getVoronoiFacetList([])
	for i in xrange(0,len(facets)) :
		ifacet_arr = []
		for f in facets[i] :
			ifacet_arr.append(f)
		
		ifacet = np.array(ifacet_arr, np.int)
		color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

		cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
		ifacets = np.array([ifacet])
		cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
		cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

cap = cv2.VideoCapture('/dev/video0')
# load the input image, resize it, and convert it to grayscale
ret, imageDest = cap.read()

imageDest = cv2.imread('TestSet_P2/Rambo.jpg')
subdivDest,image1,destPoints = getTri(imageDest)
bListInv1 = calBarycentricInv(subdivDest)

imageSource = cv2.imread('TestSet_P2/Scarlett.jpg')
subdivSrc,image2,srcPoints = getTri(imageSource)
bList2 = calBarycentric(subdivSrc)

indexesTrianglesDest = getTriIndices(subdivDest,destPoints,imageDest)
print(indexesTrianglesDest)
for triangle_index in indexesTrianglesDest:
	pt1 = srcPoints[triangle_index[0]]
	pt2 = srcPoints[triangle_index[1]]
	pt3 = srcPoints[triangle_index[2]]
	print(pt1)
	print(pt2)
	print(pt3)
	cv2.line(imageSource, pt1, pt2, (0, 0, 255), 2)
	cv2.line(imageSource, pt3, pt2, (0, 0, 255), 2)
	cv2.line(imageSource, pt1, pt3, (0, 0, 255), 2)

cv2.imshow('denauly1',imageDest)
cv2.imshow('denauly2',imageSource)
cv2.waitKey(0)


# print(imageDest.shape)
# # # Interate through destination image
# for x in range(imageDest.shape[1]):
# 	for y in range(imageDest.shape[0]):
# 		print(x,y)
# 		for bInv in bListInv1:
# 			abc = np.matmul(bInv,np.array([x,y,1]))
# 			# check if pixel is inside the triangle
# 			if(0<=abc[0]<=1):
# 				if(0<=abc[1]<=1):
# 					if(0<=(abc[0]+abc[1]+abc[2])<=1):



# # Draw delaunay triangles
# draw_delaunay(image, subdiv, (255, 255, 255) )

# cv2.imshow('voronoi',image)
# cv2.waitKey(0)