# ================================================
# Skeleton codes for HW4
# Read the skeleton codes carefully and put all your
# codes into main function
# ================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay
import sys

def help_message():
   print("Usage: [Input_Image] [Input_Marking] [Output_Directory]")
   print("[Input_Image]")
   print("Path to the input image")
   print("[Input_Marking]")
   print("Path to the input marking")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " astronaut.png " + "astronaut_marking.png " + "./")

def help_run():
   print("Press b to capture background")
   print("Press f to capture foreground")
   print("Press e to evaluate")
   print("Press n to create new canvas")

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=19)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)

def RMSD(target, master):
    # Note: use grayscale images only

    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        return -1
    else:

        total_diff = 0.0;
        dst = cv2.absdiff(master, target)
        dst = cv2.pow(dst, 2)
        mean = cv2.mean(dst)
        total_diff = mean[0]**(1/2.0)

        return total_diff;

drawing = False # true if mouse is pressed
ix,iy = -1,-1
background = False 
# mouse callback function
def enable_background():
    global background
    background = True

def enable_foreground():
    global background
    background = False

#Reference : http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode, background
    if background == True:
        sketchColor = (255,0,0)
    else:
        sketchColor = (0,0,255)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(mask_img,(x,y),5,sketchColor,-1)
            cv2.circle(img_original,(x,y),5,sketchColor,-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask_img,(x,y),5,sketchColor,-1)
        cv2.circle(img_original,(x,y),5,sketchColor,-1)

def evaluate(img_marking):

    img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img)

    fg_segments, bg_segments = find_superpixels_under_marking(img_marking, superpixels)

    fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
    bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)
    
    norm_hists = normalize_histograms(color_hists)

    graph_cut = do_graph_cut((fg_cumulative_hist, bg_cumulative_hist), (fg_segments, bg_segments), norm_hists, neighbors)

    # for c,g in zip(centers,graph_cut):
    #     if g == True:
    #         img = cv2.circle(img,(np.int32(c[1]),np.int32(c[0])),5,(0,255,120),-1)
    #     else:
    #         img = cv2.circle(img,(np.int32(c[1]),np.int32(c[0])),5,(0,120,255),-1)

    # cv2.imshow('img', img)


    mask = np.zeros_like(superpixels, np.uint8)
    for i in enumerate(graph_cut):
            if graph_cut[i] == True:
                mask[superpixels == i[0]] = 255
    
    cv2.imshow('mask', mask)

def createnewcanvas():
    global img_original, mask_img
    img_original = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    mask_img = np.zeros_like(img_original, np.uint8)
    mask_img.fill(255)


if __name__ == '__main__':
 
    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    help_run()
    imagePath = sys.argv[1]   
    img_original = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    mask_img = np.zeros_like(img_original, np.uint8)
    mask_img.fill(255)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image',img_original)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('b'):               #Change to capture background
            enable_background()
        if k == ord('f'):               #Change to capture foreground
            enable_foreground()
        elif k == ord('e'):             #Evaluate
            evaluate(mask_img)
        elif k == ord('n'):             #Create New Canvas
            createnewcanvas()
        elif k == 27:
            break

    cv2.destroyAllWindows()


