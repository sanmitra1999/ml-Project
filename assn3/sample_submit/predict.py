import numpy as np
import segment
import model

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that 
# were given. The evaluation code may give unexpected results if this convention is not followed.
def int2char(i):
	return chr(ord('A')+i)


def decaptcha( filenames ):
	numChars = 3 * np.ones( (len( filenames ),) )
	codes = []
	all_chars = []
	for j,file in enumerate(filenames):
		chars = segment.segment_image(file)
		all_chars.append(chars)
		numChars[j]=len(chars) 
	all_ans = model.pred(all_chars)
	for i in all_ans:
		ans=''
		for j in i:
			ans+=int2char(j)
		codes.append(ans)
	return (numChars, codes)