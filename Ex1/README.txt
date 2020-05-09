Ex_1 - omer paz 
id: 311326490


infomation:

------------------------------------------------------------------------
Python Version: 3.8.2
------------------------------------------------------------------------

files attached:

ex1_utils.py - all the function that we had to implements in the ex_1.
gamma.py - one function that made tracker bar to gamma correction between 0.01 -> 2
testImage1.jpg - this image if from the university day, i took this picture and all the people there in black beacue of the show.
	         so i pick this picture cuse i want to see if i can see some faces in the histogram eq function.
testImage2.jpg - this picutre took and its look alot of black, so i whant to see our faces with a good contras option.. then i took this pic
 		 and i hope to see our faces in the results in histogram eq.

------------------------------------------------------------------------

functions and description in ex1_utils.py gamma.py:

1.imReadAndConvert: read image grayscle/rgb and convert it to array normalize to [0,1].
2. normalize: normalize the image between [0,1].
3. unnormalize: unnormalize the the image between [0,255]. 
4. imDisplay: disply the image, using matplotlib.pyplot.
5. transformRGB2YIQ: transfrom image from RGB to YIQ color space.
6. transformYIQ2RGB: transfrom image from YIQ to RGB color space.
7. hsitogramEqualize: get the Equalize hsitogram for given image called: imgEQ
8. quantizeImage: this function will quantize Image to nQuant parts, and its will be nIter iteration.
9. (in gamma.py) gammaDisplay: will be disply tracker bar and image, and the user can change the image gamma and see the diffrent.

------------------------------------------------------------------------

answer 4.5 - its will be devision by zero, example: if we have image with no black value, i mean intensity[0-10] = 0 , then we will get devision by zero and the program crash

