# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_bonus = cv2.imread("images/pattern001.jpg", cv2.IMREAD_COLOR)
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region

    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h,w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)

        # TODO: populate scan_bits by putting the bit_code according to on_mask
        scan_bits[on_mask == True] |= bit_code

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    corresp_image = np.zeros((960,1280,3), np.uint8)
    colormesh = []
    camera_points = []
    projector_points = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code

            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points
            x_p = binary_codes_ids_codebook[scan_bits[y,x]][0]
            y_p = binary_codes_ids_codebook[scan_bits[y,x]][1]
            if x_p >= 1279 or y_p >= 799: # filter
                continue

            camera_points.append((x/2.0,y/2.0))
            projector_points.append((x_p,y_p))
            colormesh.append((ref_bonus[y][x][2], ref_bonus[y][x][1], ref_bonus[y][x][0]))
            corresp_image[y,x][2] = x_p*255/(x_p+y_p)
            corresp_image[y,x][1] = y_p*255/(x_p+y_p)
            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2

    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
    #Reference: https://programtalk.com/python-examples/cv2.undistortPoints/
    camera_uv = np.array(camera_points, dtype=np.float32) 
    cam_pts = camera_uv.size / 2 
    camera_uv.shape = (cam_pts, 1, 2)
    # print camera_uv
    camera_norm = cv2.undistortPoints(camera_uv, camera_K, camera_d,P=camera_K)
    # print 'camera_norm',camera_norm

    proj_uv = np.array(projector_points, dtype=np.float32) 
    proj_pts = proj_uv.size / 2 
    proj_uv.shape = (proj_pts, 1, 2)
    # print proj_uv.shape
    proj_norm = cv2.undistortPoints(proj_uv, projector_K, projector_d,P=projector_K)
    # print 'proj_norm',proj_norm


    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    P0 = np.dot(camera_K, np.array([ [1,0,0,0],
                [0,1,0,0],
                [0,0,1,0]   ]))

    P1 = np.concatenate((np.dot(projector_K,projector_R),np.dot(projector_K,projector_t)), axis = 1)
    triang_res = cv2.triangulatePoints(P0, P1, camera_norm, proj_norm)


    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    points_3d = cv2.convertPointsFromHomogeneous(triang_res.T)


	# TODO: name the resulted 3D points as "points_3d"
    count = 0
    color_3d = []
    mask_3d = []
    for i in range(points_3d.shape[0]):
        if (points_3d[i][0][2] > 200) & (points_3d[i][0][2] < 1400):
            mask_3d.append((round(points_3d[i][0][0]), round(points_3d[i][0][1]), round(points_3d[i][0][2])))
            color_3d.append((colormesh[i]))
            count = count + 1	

    # print 'count', count
    ans = np.array(mask_3d)
    final_3d = ans.reshape(ans.shape[0],-1).reshape(ans.shape[0],1,3)
    # print final_3d.shape

    #################################################################################################
    #Bonus Question Enable
    bonus(corresp_image, final_3d, color_3d)
    #################################################################################################

    return final_3d
	
def write_3d_points(points_3d):

    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))

    return

def bonus(corresp_image, points_3d, color_3d):
    
    # ===== DO NOT CHANGE THIS FUNCTION =====
    
    output_name =  sys.argv[1] + "correspondence.jpg"
    cv2.imwrite(output_name, corresp_image)

    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output_color.xyz"
    with open(output_name,"w") as f:
        for p,c in zip(points_3d, color_3d):
            f.write("%d %d %d %d %d %d\n"%(p[0,0],p[0,1],p[0,2],c[0],c[1],c[2]))

    return
    
if __name__ == '__main__':

	# ===== DO NOT CHANGE THIS FUNCTION =====
	
	# validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
	
