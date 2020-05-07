

import cv2.cv as cv #Import functions from OpenCV





#first read image in gray scale.
GrayImg = cv.LoadImageM("cat.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)

SmoothGrayImg = cv.CreateMat(GrayImg.rows, GrayImg.cols, cv.CV_BLUR_NO_SCALE)
cv.Smooth(GrayImg, SmoothGrayImg  ,cv.CV_MEDIAN, param1=5, param2=0, param3=0, param4=0)
cv.SaveImage("Grey image.png", GrayImg)
cv.SaveImage("GraySmooth image.png",SmoothGrayImg )
#edge detection
EdgeDetection_Img = cv.CreateImage(cv.GetSize(SmoothGrayImg ), cv.IPL_DEPTH_16S, cv.CV_BLUR_NO_SCALE)
cv.Laplace(SmoothGrayImg , EdgeDetection_Img )
cv.SaveImage(" EdgeDetection image.png",EdgeDetection_Img )
# set threshold
Thresholding = cv.CreateImage(cv.GetSize(EdgeDetection_Img ), cv.IPL_DEPTH_16S, cv.CV_BLUR_NO_SCALE)
cv.Threshold(EdgeDetection_Img , Thresholding,20, 400, cv.CV_THRESH_BINARY_INV)
cv.SaveImage("Thresholding.png",Thresholding)

#Output from bilateral filter
im = cv.LoadImageM("cat.jpg")
BilateralImg=cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_8U, 3);
cv.CvtColor(im,BilateralImg , cv.CV_RGB2Lab); 


cv.Smooth(im, BilateralImg, cv.CV_BILATERAL, 100, 100, 100,100); 
cv.SaveImage("bilateral.png",BilateralImg)
finalImg=cv.CreateImage(cv.GetSize(GrayImg), cv.IPL_DEPTH_8U, 3);

Sketch= cv.LoadImageM("Thresholding.png")
Paint = cv.LoadImageM("bilateral.png")
cv.And(Sketch, Paint,finalImg)
cv.SaveImage("final.png",finalImg )



 