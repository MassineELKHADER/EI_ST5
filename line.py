from __future__ import division
import cv2
import numpy as np

detect_inter = 0
# Charger l'image
image = cv2.imread("photo_carrefour1.jpg")

# Obtenir les dimensions de l'image
h, w = image.shape[:2]
print(h,w)
# Coordonnées des points pour former un triangle
#triangle_points = np.array([[0, h], [w, h], [w // 2, h //2  ]], dtype=np.int32)
# Convertir en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Définir la ROI pour le triangle
#roi = np.zeros_like(gray)
#cv2.fillPoly(roi, [triangle_points], 255)

# Appliquer le seuillage à la ROI
ret, thresh1 = cv2.threshold(blurred, 175, 255, cv2.THRESH_BINARY)
#roi = cv2.bitwise_and(thresh1, roi)

# Trouver les contours dans la ROI
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Calculer les coordonnées du coin supérieur gauche du carré

square_size = min(w, h) // 100

# Dessiner les contours uniquement dans le triangle
contour_image = np.zeros_like(image)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Trier les contours par surface (garder seulement le plus grand)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

if len(contours) > 0:
    M = cv2.moments(contours[0])
    # Centroid
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print("Centroid of the biggest area: ({}, {})".format(cx, cy))
else:
    print("No Centroid Found")
center_x, center_y = cx, cy
top_left_x = center_x - square_size // 2
top_left_y = center_y - square_size // 2
cv2.rectangle(contour_image, (top_left_x, top_left_y), (top_left_x + square_size, top_left_y + square_size), (0, 0, 255), 2)

edges = cv2.Canny(thresh1, 50, 150)
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
list_lines =[]
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    list_lines.append([(x1, y1),(x2, y2)])
    cv2.line(contour_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

# Enregistrer l'image résultante
def calculer_angle(ligne1, ligne2):
    delta_x1 = ligne1[1][0] - ligne1[0][0]
    delta_y1 = ligne1[1][1] - ligne1[0][1]

    delta_x2 = ligne2[1][0] - ligne2[0][0]
    delta_y2 = ligne2[1][1] - ligne2[0][1]
    
    # Utilisation d'arctan2 pour calculer l'angle entre les deux lignes
    angle_rad = abs(np.arctan2(delta_y2, delta_x2)) - abs(np.arctan2(delta_y1, delta_x1))

    # Conversion de radians à degrés
    angle_deg = np.degrees(angle_rad)
    # Assurer que l'angle est dans la plage [0, 90)
    angle_deg = abs(angle_deg) % 90
    
    return angle_deg

def lignes_ont_angle_suffisant(ligne1, ligne2, seuil_angle):
    angle = calculer_angle(ligne1, ligne2)
    return angle > seuil_angle

for i in range(1,len(list_lines)):
    line_halal = list_lines[0]
    if lignes_ont_angle_suffisant(list_lines[i],line_halal,seuil_angle=80):
        print('SALOPE YOUNES')
        detect_inter = 1
        break
    else:
        print('MERDE')
        detect_inter = 0 
        


    

    


cv2.imwrite('out_test.png', contour_image)
