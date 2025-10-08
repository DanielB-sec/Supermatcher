import cv2

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

def extract_minutiae(binary_image):
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  minutiae = []
  for contour in contours:
    if cv2.contourArea(contour) > 50: # Filter small contours
      for point in contour:
        x, y = point[0]
        minutiae.append((x, y))
  return minutiae

def match_fingerprints(minutiae1, minutiae2):
  match_score = abs(len(minutiae1) - len(minutiae2))
  return match_score

image_path1 = "101_1.tif"
#image_path2 = "101_2.tif"
binary_image1 = preprocess_image(image_path1)
#binary_image2 = preprocess_image(image_path2)
minutiae1 = extract_minutiae(binary_image1)
#minutiae2 = extract_minutiae(binary_image2)

import os
for file in os.listdir("."):
  #print("testing file", file)
  try:
    binary = preprocess_image(file)
    minutiae = extract_minutiae(binary)
    match_score = match_fingerprints(minutiae1, minutiae)
    threshold = 100
    if match_score < threshold:
      print(file, "Fingerprints match!", match_score)
    else:
      print(file, "Fingerprints do not match.", match_score)
  except Exception as e:
    pass #print(e)
