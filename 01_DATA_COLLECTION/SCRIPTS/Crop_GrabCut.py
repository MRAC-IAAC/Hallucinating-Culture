import cv2
import glob
import numpy as np

#Open & Save multiples images
#https://github.com/sanpreet/Write-Multiple-images-into-a-folder-using-python-cv2

#GrabCut
#https://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html

#ROI selection
#https://learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/


path = "G:/Mi unidad/01_TERM II - GROUP WORK FOLDER/HARDWARE II - GROUP WORK FOLDER/05_Scripts/columns/*.*"
for index,file in enumerate (glob.glob(path)):
    i_im = cv2.imread(file)

# Resize Image
    width = 468
    height = 702
    dim = (width, height)

    im = cv2.resize(i_im, dim, 0, 0, cv2.INTER_AREA)

########

    r = cv2.selectROI(im)
    imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    mask = np.zeros(im.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = r
    cv2.grabCut(im, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Apply the above mask to the image
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = im * mask2[:, :, np.newaxis]

    # Get the background
    background = im - img

    # Change all pixels in the background that are not black to white
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    # Add the background and the image
    final = background + img

########
    # Display the image
    cv2.imshow("image", final)
    # writing the images in a folder output_images
    cv2.imwrite('G:/Mi unidad/01_TERM II - GROUP WORK FOLDER/HARDWARE II - GROUP WORK FOLDER/05_Scripts/newcolumns/00{}.png'.format(index), final)
    # wait for 1 second
    cv2.waitKey(0)
    # destroy the window
    cv2.destroyAllWindows()