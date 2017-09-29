# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np

def CDF(hist):
   cdfArray = hist
   for x in xrange(1,len(cdfArray)):
      cdfArray[x] += cdfArray[x-1]
   return cdfArray

#Reference : http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
def monochrome(img_in):
   hist,bins = np.histogram(img_in,256,[0,256])
   cdf = CDF(hist)   
   cdf = (cdf*255)/(cdf.max()-cdf.min())
   img_out = cdf[img_in].astype('uint8')
   return img_out 

#Reference:https://piazza.com/class/j6vr5jgnh9d366?cid=55
def ft(im, newsize=None):
    dft = np.fft.fft2(np.float32(im),newsize)
    return np.fft.fftshift(dft)

#Reference:https://piazza.com/class/j6vr5jgnh9d366?cid=55
def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):
   cv2.imshow('In', img_in)
   # Write histogram equalization here
   b,g,r = cv2.split(img_in)

   b2 = monochrome(b)                     #Call histogram equalization for each band
   g2 = monochrome(g)
   r2 = monochrome(r)

   img_out = cv2.merge((b2,g2,r2))
   return True, img_out
   
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);

   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================
#Reference:https://piazza.com/class/j6vr5jgnh9d366?cid=55, http://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html
def low_pass_filter(img_in):
	
   # Write low pass filter here
   imf = ft(img_in, (img_in.shape[0],img_in.shape[1])) 
   mask = np.zeros((img_in.shape[0],img_in.shape[1]),np.uint8)                              #Box Filter of 20X20 size
   mask[img_in.shape[0]/2-20:img_in.shape[0]/2+20, img_in.shape[1]/2-20:img_in.shape[1]/2+20] = 1
   img_out = ift(imf*mask).astype('uint8')
   return True, img_out

#Reference:https://piazza.com/class/j6vr5jgnh9d366?cid=55, http://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html
def high_pass_filter(img_in):

   # Write high pass filter here
   imf = ft(img_in, (img_in.shape[0],img_in.shape[1])) 
   mask = np.ones((img_in.shape[0],img_in.shape[1]),np.uint8)                            #Box Filter of 20X20 size
   mask[img_in.shape[0]/2-20:img_in.shape[0]/2+20, img_in.shape[1]/2-20:img_in.shape[1]/2+20] = 0
   img_out = ift(imf*mask).astype('uint8')  
   return True, img_out
   
#Reference:https://piazza.com/class/j6vr5jgnh9d366?cid=55, http://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html
def deconvolution(img_in):
   
   # Write deconvolution codes here
   gk = cv2.getGaussianKernel(21,5)
   gk = gk * gk.T
   imf = ft(img_in, (img_in.shape[0],img_in.shape[1])) # make sure sizes match
   gkf = ft(gk, (img_in.shape[0],img_in.shape[1]))
   img_out = ift(imf/gkf)*255
   return True, img_out

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

#Reference: http://docs.opencv.org/3.2.0/dc/dff/tutorial_py_pyramids.html
def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   G = img_in1.copy()
   R = img_in2.copy()

   G = G[:,:G.shape[0]]                   #make images rectangular and equal size
   R = R[:G.shape[0],:G.shape[0]]

   #Gaussian Blur and downsample
   gausG = [G]
   gausR = [R]
   for i in xrange(6):
      G = cv2.pyrDown(G)
      gausG.append(G)
      R = cv2.pyrDown(R)
      gausR.append(R)

   # Laplacian pyramid
   lowpassG = [gausG[5]]
   lowpassR = [gausR[5]]
   for i in xrange(5,0,-1):
      lpG = cv2.pyrUp(gausG[i], dstsize=(gausG[i-1].shape[0], gausG[i-1].shape[1])) #Upscale the blur image so that its dimension matches with non blur image in previous pyramid level
      highFreqG = cv2.subtract(gausG[i-1],lpG)                                      #Subtract non blur and blur image to get high frequency image
      lowpassG.append(highFreqG)

      lpR = cv2.pyrUp(gausR[i], dstsize=(gausR[i-1].shape[0], gausR[i-1].shape[1]))
      highFreqR = cv2.subtract(gausR[i-1],lpR)
      lowpassR.append(highFreqR)

   BlendGR = []
   for lG,lR in zip(lowpassG,lowpassR):
      lGR = np.hstack((lG[:,0:lG.shape[1]/2], lR[:,lR.shape[1]/2:]))                #Take left portions of Green image and right portion of Red image and make a new image
      BlendGR.append(lGR)


   img_out = BlendGR[0]
   for i in xrange(1,6):
      img_out = cv2.pyrUp(img_out, dstsize=(BlendGR[i].shape[0], BlendGR[i].shape[1])) #Move up the pyramid and add
      img_out = cv2.add(img_out, BlendGR[i])

   return True, img_out

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
	  
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
