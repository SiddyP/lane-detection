import numpy as np
from PIL import ImageGrab
import cv2
import time
from numpy import ones,vstack
from numpy.linalg import lstsq
from directkeys import PressKey, W, A, S, D
from statistics import mean
from matplotlib import pyplot as plt
import math

def roi(img, vertices):
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, vertices, 255)
	masked = cv2.bitwise_and(img, mask)
	return masked

def draw_lanes(img, lines, color=[255,100,255], thickness=3):
	try:
		pt1 = []
		pt2 = []
		line_dict_neg = {}
		line_dict_pos = {}
		for i,line in enumerate(lines):
			x_coords = (line[0][0],line[0][2])
			y_coords = (line[0][1],line[0][3])

			A = vstack([x_coords, ones(len(x_coords))]).T
			m, b = lstsq(A, y_coords)[0]
			if m < -0.2:
				dist = math.hypot(x_coords[1]-x_coords[0], y_coords[1]-y_coords[0])
				pt1.append((x_coords[0], y_coords[0]))
				pt2.append((x_coords[1], y_coords[1]))
				line_dict_neg[i] = [dist,(x_coords[0], y_coords[0]),(x_coords[1], y_coords[1])]
			if m > 0.2:
				dist = math.hypot(x_coords[1]-x_coords[0], y_coords[1]-y_coords[0])
				pt1.append((x_coords[0], y_coords[0]))
				pt2.append((x_coords[1], y_coords[1]))
				line_dict_pos[i] = [dist,(x_coords[0], y_coords[0]),(x_coords[1], y_coords[1])]

		max_dist = 0
		for items in line_dict_neg:
			dist = line_dict_neg[items][0] 
			if dist > max_dist:
				max_dist = dist
				neg_index = items


		max_dist = 0
		for items in line_dict_pos:
			dist = line_dict_pos[items][0]
			if dist > max_dist:
				max_dist = dist
				pos_index = items

		point1 = []
		point2 = []

		point1.append(line_dict_neg[neg_index][1])
		point1.append(line_dict_pos[pos_index][1])

		point2.append(line_dict_neg[neg_index][2])
		point2.append(line_dict_pos[pos_index][2])
		print([point1[0][0], point1[0][1], point1[1][0], point1[1][1], point2[0][0], point2[0][1], point2[1][0], point2[1][1]])
		return [point1[0][0], point1[0][1], point2[0][0], point2[0][1]], [point1[1][0], point1[1][1], point2[1][0], point2[1][1]]
	except:
		pass


def process_img(original_img):
	processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
	processed_img = cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
	vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],], np.int32)

	processed_img = cv2.GaussianBlur(processed_img, (3,3),0)
	processed_img = roi(processed_img, [vertices])
	lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 20, 15)
	try:
		l1, l2 = draw_lanes(original_img, lines)
		cv2.line(original_img, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 5)
		cv2.line(original_img, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 5)
	except Exception as e:
		print(str(e))
		pass
	try:
		for coords in lines:
			coords = coords[0]
			try:
				cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)

			except Exception as e:
				print(str(e))
	except Exception as e:
		pass
	return processed_img, original_img


def main(): 
	last_time = time.time()
	while(True):
		screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
		#print(f'Loop took {time.time()-last_time} seconds')
		last_time = time.time()
		new_screen, original_img = process_img(screen)
		cv2.imshow('window', new_screen)
		cv2.imshow('window2',cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

		#cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
main()


