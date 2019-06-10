import imutils
 
def pyramid(image, downscale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
 
	# loop over the pyramid yielding smaller images
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / downscale)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the given minimum size, 
		# stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image

# METHOD #2: Resizing + Gaussian smoothing.
from skimage.transform import pyramid_gaussian as _pyramid_gaussian

def pyramid_gaussian(image, downscale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	for (i, resized) in enumerate(_pyramid_gaussian(image, downscale=downscale)):

		# if the image is too small, break from the loop
		if resized.shape[0] < minSize[0] or resized.shape[1] < minSize[1]:
			break
		
		# yield the next image in the pyramid
		yield resized
