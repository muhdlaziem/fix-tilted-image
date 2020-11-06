import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import os
import argparse

"""
    Function get from https://stackoverflow.com/questions/57964634/python-opencv-skew-correction-for-ocr
"""
def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

def saveImage(path, image, rotated, angle):
    print(f'[INFO] Angle : {angle}, size original : {image.shape}, size rotated : {rotated.shape}')
    imageName = path.split('/')[-1]
    path = f'outputs/{imageName}_rotated.png'
    if(os.path.exists("outputs")):

        if(os.path.exists("test")):
            cv2.imwrite(f'test/{imageName}.png', image)
        else:
            os.mkdir("test")
            cv2.imwrite(f'test/{imageName}.png', image)

        cv2.imwrite(path, rotated)
        print(f'[INFO] Saved to {path}')

    else:
        os.mkdir("outputs")

        if(os.path.exists("test")):
            cv2.imwrite(f'test/{imageName}.png', image)
        else:
            os.mkdir("test")
            cv2.imwrite(f'test/{imageName}.png', image)
            
        cv2.imwrite(path, rotated)
        print(f'[INFO] Saved to {path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="image path",type=str)
    args = parser.parse_args()
   
    if(args.p.endswith(".tif")):
        images =  cv2.imreadmulti(args.p)
        images = images[1]
        extracted_images = []
        for i, image in enumerate(images):
            img = np.asarray(image)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            # print(img.shape)
            angle, rotated = correct_skew(img)
            # print(f'{args.p}_{i}')
            saveImage(f'{args.p}_{i}',img, rotated, angle)

    else:
        image = cv2.imread(args.p)
        angle, rotated = correct_skew(image)

        saveImage(args.p, image, rotated, angle)
        
