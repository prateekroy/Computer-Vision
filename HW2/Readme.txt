HW2: Image Alignment, Panoramas
 
Your goal is to create 2 panoramas:
Using homographies and perspective warping on a common plane (3 images).
Using cylindrical warping (many images).
In both options you should:
Read in the images: input1.jpg, input2.jpg, input3.jpg
[Apply cylindrical wrapping if needed]
Calculate the transformation (homography for projective; affine for cylindrical) between each
Transform input2 and input3 to the plane of input1, and produce output.png
Bonus (!!): Use your Laplacian Blending code to stitch the images together nicely