"""
Character Detection

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import copy

import task1
import utils
import numpy as np
import cv2


  # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/proj1-task2-png.png",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="./data/c.jpg",
        choices=["./data/a.jpg", "./data/a.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def detect(img, template):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # TODO: implement this function.

    det=corr_func(img, template)
    count=0
    list1=[]
    
  
    threshold=0.85
    
    for i,row in enumerate(det):
        for j,num in enumerate(row):
            if(det[i][j]>threshold):
                count=count+1
                pos=(i,j)
                list1.append(pos)      
    print("Count",count)
    print(list1)
    
    
    #raise NotImplementedError()
    return list1

def corr_func(img,template):
    r1=len(img)
    c1=len(img[0])
    r2=len(template)
    c2=len(template[0])
    rows=r1-r2+1
    cols=c1-c2+1
    
    corr=np.ones(shape=(rows,cols))
    
    val1=subtract(template,cal_mean(template))
    meanTemplate_Square=np.multiply((val1), val1)
    meanTemplateSquare_Sum=addAllElements(meanTemplate_Square)
    
    
    for i in range(rows):
        for j in range(cols):
            cropped_img=utils.crop(img, i, i+r2, j, j+c2)
            cropped_img_mean=subtract(cropped_img,cal_mean(cropped_img))
            
            numerator=task1.multiply_add(cropped_img_mean, val1)
            crop_sum=task1.multiply_add(cropped_img_mean, cropped_img_mean)
            mulValue=crop_sum*meanTemplateSquare_Sum
            denominator=mulValue**(1/2)
            if(denominator!=0):
                corr[i][j]=numerator/denominator
            else:
                corr[i][j]=0
    return corr
    

def task(img,kernel):
    accum = 0
    img_conv = copy.deepcopy(img)
    for m, row in enumerate(img):
        for i, pixel in enumerate(row):
            cp = utils.crop(img,m,m+3,i,i+3)
            mul = utils.elementwise_mul(cp, kernel)
            for k, row in enumerate(mul):
                for j, num in enumerate(row):
                    accum = accum + mul[k][j]
            img_conv[m][i] = accum
            accum = 0
            
    return img_conv

def addAllElements(a):
    sum=0
    for i,row in enumerate(a):
        for j,num in enumerate(row):
            sum=sum+a[i][j]
    
    return sum
    


def cal_mean(mat):
    count=0
    sum=0
    for i,row in enumerate(mat):
        for j,num in enumerate(row):
            sum=sum+mat[i][j]
            count=count+1
    
    mean=sum/count
    return mean

def subtract(mat,val):
    for i,row in enumerate(mat):
        for j,num in enumerate(row):
            mat[i][j]=mat[i][j]-val
    
    return mat


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    


    img = task1.read_image(args.img_path)
    template = task1.read_image(args.template_path)
    
    

    coordinates = detect(img, template)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)
    
    img = cv2.imread(args.img_path,1)
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arrow, Circle
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for i in coordinates:
        ax.add_patch(Circle((i[1], i[0]), radius=1, color='red'))
    plt.show(fig)
if __name__ == "__main__":
    main()
