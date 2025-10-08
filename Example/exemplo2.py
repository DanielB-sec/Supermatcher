import cv2
import numpy as np

def preprocess_image(file):
  image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
  #cv2.imshow("grayscale", image)
  image = cv2.equalizeHist(image)
  #cv2.imshow("histogram", image)
  image = cv2.GaussianBlur(image, (5, 5), 0)
  #cv2.imshow("blur", image)
  _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  #cv2.imshow("binary", binary_image)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()
  return binary_image


def euclidean_distance(point1, point2):
  return np.linalg.norm(np.array(point1) - np.array(point2))

def match_fingerprints(minutiae1, minutiae2):
  distances = np.zeros((len(minutiae1), len(minutiae2)))
  for i, m1 in enumerate(minutiae1):
    for j, m2 in enumerate(minutiae2):
      distances[i, j] = euclidean_distance(m1, m2)
  min_distances = np.min(distances, axis=1)
  similarity_score = np.mean(min_distances)
  return similarity_score

def extract_minutiae(binary_image):
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  minutiae = []
  for contour in contours:
    if cv2.contourArea(contour) > 50: # Filter small contours
      for point in contour:
        x, y = point[0]
        minutiae.append((x, y))
  return minutiae

image_path1 = "101_1.tif"
#image_path2 = "101_2.tif"
binary_image1 = preprocess_image(image_path1)
#binary_image2 = preprocess_image(image_path2)
minutiae1 = extract_minutiae(binary_image1)
#minutiae2 = extract_minutiae(binary_image2)

import os
for file in os.listdir("."):
  try:
    binary = preprocess_image(file)
    minutiae = extract_minutiae(binary)
    match_score = match_fingerprints(minutiae1, minutiae)
    threshold = 5
    if match_score < threshold:
      print(file, "Fingerprints match!", match_score)
    else:
      print(file, "Fingerprints do not match!", match_score)
  except Exception as e:
    print(e)
