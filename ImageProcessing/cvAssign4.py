
import cv2
import matplotlib.pyplot as plt
import numpy as np

from numpy.linalg.linalg import dot,inv
from matplotlib.pyplot import plot
from numpy.linalg.linalg import lstsq
import  scipy.interpolate

#2 Computing the homography parameters
def computeH(pts1,pts2):
    l1, l2 = len(pts1), len(pts2)
    assert l1 == l2
    
    A=[]
    B=[]
    for j in range(l1*2):
        i=int(j/2)
        if(j%2==0):
            A.append([pts1[i][0],pts1[i][1],1,0,0,0,-pts2[i][0]*pts1[i][0],-pts2[i][0]*pts1[i][1]])
        else:
            A.append([0,0,0,pts1[i][0],pts1[i][1],1,-pts2[i][1]*pts1[i][0],-pts2[i][1]*pts1[i][1]])

    for i in range(l2):
        B.append(pts2[i][0])
        B.append(pts2[i][1])
        
    # H matrix 3*3
    a,b,c,d,e,f,g,h = lstsq(A, B)[0]
    
    return [[a,b,c],[d,e,f],[g,h,1]]
 #Getting correspondences
   
def getCorrespondence(imageA, imageB , num):
    fig = plt.figure()
    figA = fig.add_subplot(1,2,1)
    figB = fig.add_subplot(1,2,2)
    figA.imshow(imageA,origin='upper')
    figB.imshow(imageB,origin='upper')
    plt.axis('image')
    pts = plt.ginput(n=num*2, timeout=0)
    pts = np.reshape(pts, (2, pts, 2))
    return pts[0],pts[1];



 #   3 Warping between image planes

    
def warping2Images(H , image , image2):
    H_img = image.shape[0];
    W_img = image.shape[1];

    
    # get the boundaries point 
    Point1=np.dot(H,np.array([0,0,1]));
    Point1= Point1/ Point1[2];  
    Point2=np.dot(H,np.array([W_img-1,0,1]));
    Point2= Point2/ Point2[2];
    Point3=np.dot(H,np.array([0,H_img-1,1]));
    Point3=Point3/Point3[2];
    Point4=np.dot(H,np.array([W_img-1,H_img-1,1]));
    Point4=Point4/Point4[2];
     
    dim1=np.shape(image);
    height2=dim1[0];
    width2=dim1[1];   
    
    
    # get the exact location for the point (x,y)
    minX = min( Point1[0], Point2[0],Point3[0],Point4[0]);
    minX=min(minX,0);

    maxX = max( Point1[0], Point2[0],Point3[0],Point4[0]);
    maxX=max(maxX,width2);

    minY = min( Point1[1], Point2[1],Point3[1],Point4[1]);
    minY=min(minY,0);

    maxY = max( Point1[1], Point2[1],Point3[1],Point4[1]);
    maxY=max(maxY,height2);
    
    shift_V = 0;
    shift_H = 0;
    if(minY < 0):
        shift_V = -minY;
    if(minX < 0):
        shift_H = -minX;
    print shift_V;
    print shift_H;
    warped_image = np.zeros((maxY+shift_V , maxX+shift_H,3), dtype=np.double);
    HInverse = np.linalg.inv(H);
    # Interpolation
    hh = np.arange(H_img)
    ww= np.arange(W_img)
    
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
   
    red_container = scipy.interpolate.RectBivariateSpline(hh, ww, red, kx=2, ky=2);
    green_container = scipy.interpolate.RectBivariateSpline(hh, ww, green, kx=2, ky=2);
    blue_container = scipy.interpolate.RectBivariateSpline(hh, ww, blue, kx=2, ky=2);
    
    for i in range(int(minY), int(maxY)):
        for j in range( int(minX), int(maxX)):
            temp_point = np.array([j,i,1]);
            point = np.dot(HInverse, temp_point);
            point = point/point[2];
            x=int(point[1]);
            y=int(point[0]);

            warped_image[i+ shift_V][j+shift_H][0]=int(red_container.ev(x , y));
            warped_image[i+ shift_V][j+shift_H][1]=int(green_container.ev(x , y));
            warped_image[i+ shift_V][j+shift_H][2]=int(blue_container.ev(x , y));

  
    h2,w2,d2 = image2.shape;
    
    for i in range(h2):
        for j in range(w2):
            warped_image[int(i+shift_V)][int(j+shift_H)]=(image2[i][j]);
    return warped_image;

def Verify(H,pts):
    len= np.shape(pts)[0]
    for i in range(0,len):
        p=np.zeros((3,1), dtype=np.double);
        p[0]=pts[i][0]
        p[1]=pts[i][1]
        p[2]=1

        newp= np.dot(H, p)
        newp=newp/newp[2]
    
        print 'point('+str(pts[0])+','+str(pts[1])+') ---map to---> ('+str(newp[0])+','+str(newp[1])+')'




if __name__ == "__main__":

    imageA = cv2.imread('uttower1.jpg')
    imageB = cv2.imread('uttower2.jpg')
    
    pts1,pts2 = getCorrespondence(imageA, imageB,4)
    print pts1
    print pts2

    H =computeH(pts1,pts2)
    print H
    
    Verify(H,pts1)



warpedImage = warping2Images(H, imageA , imageB);

cv2.imwrite("warpedImage.jpg", warpedImage);
cv2.imshow('warped Image', warpedImage)
cv2.waitKey(0)
    
    
    
    
