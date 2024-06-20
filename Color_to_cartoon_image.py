import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def read_file(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()
    return img

def edge_mark(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape((img.shape))
    return result

def cartoon():
    c = cv2.bitwise_and(blurred, blurred, mask=edges)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(org_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(c)
    plt.title("Cartoon Image")
    plt.axis('off')
    
    plt.show()

# Use tkinter to choose a file
root = Tk()
root.withdraw()  # Hide the root window
filename = askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
root.update()

if filename:
    img = read_file(filename)
    org_img = np.copy(img)

    line_size, blur_value = 7, 7
    edges = edge_mark(img, line_size, blur_value)
    
    plt.imshow(edges, cmap="gray")
    plt.title("Edge Marking")
    plt.axis('off')
    plt.show()

    img = color_quantization(img, k=9)
    
    plt.imshow(img)
    plt.title("Color Quantization")
    plt.axis('off')
    plt.show()

    blurred = cv2.bilateralFilter(img, d=3, sigmaColor=200, sigmaSpace=200)
    
    plt.imshow(blurred)
    plt.title("Blurred Image")
    plt.axis('off')
    plt.show()

    cartoon()
else:
    print("No file selected.")
