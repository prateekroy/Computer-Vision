HW4: Segmentation

Your goal is to perform semi-automatic binary segmentation based on SLIC superpixels and graph-cuts:
	Given an image and sparse markings for foreground and background
	Calculate SLIC over image
	Calculate color histograms for all superpixels
	Calculate color histograms for FG and BG
	Construct a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term)
	Run a graph-cut algorithm to get the final segmentation
 
"Wow factor" bonus (20pt):
	Make it interactive: Let the user draw the markings (carrying 0 pt for this part)
	for every interaction step (mouse click, drag, etc.)
	recalculate only the FG-BG histograms,
	construct the graph and get a segmentation from the max-flow graph-cut,
	show the result immediately to the user (should be fast enough).


Run command:
python main.py astronaut.png astronaut_marking.png ./
python main_bonus.py astronaut.png