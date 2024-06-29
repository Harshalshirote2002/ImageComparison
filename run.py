import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Path to the database folder
database_path = "dataBase"
# Path to query image
queryImagePath = "query.jpg"

imageB = cv2.imread(queryImagePath)

# Mask to exclude white background pixels [255, 255, 255]
maskB = cv2.inRange(imageB, np.array([0, 0, 0]), np.array([255, 255, 254]))

histB = cv2.calcHist([imageB], [0, 1, 2], maskB, [8, 8, 8], [0, 256, 0, 256, 0, 256])
histB = cv2.normalize(histB, histB).flatten()

info = []
for filename in os.listdir(database_path):
    filepath = os.path.join(database_path, filename)
    imageA = cv2.imread(filepath)
    maskA = cv2.inRange(imageA, np.array([0, 0, 0]), np.array([255, 255, 254]))
    histA = cv2.calcHist(
        [imageA], [0, 1, 2], maskA, [8, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    histA = cv2.normalize(histA, histA).flatten()
    info.append(
        {
            "filename": f"{filename}",
            "corr": cv2.compareHist(
                histA, histB, cv2.HISTCMP_CORREL
            ),  # Compare the normalized flattened histograms of current image and query Image using Correlation method
            "hist": histA,
        }
    )

# Sort the objects using correlation value as key in decreasing order
info.sort(key=lambda x: x["corr"], reverse=True)

print(
    f"Best Match for Query Image: {info[0]['filename']} with correlation: {info[0]['corr']}"
)
