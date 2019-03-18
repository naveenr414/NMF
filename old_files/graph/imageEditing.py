from PIL import Image
import numpy as np

def getLocation(obj,angle):
    """Find the location of the image for obj/angle pair"""
    imageName = "obj"+str(obj)+"__"+str(angle)+".png"
    return "coil-20-proc/"+imageName

def shrinkImages():
    """Shrink COIL20 Image data set
    The data set is a set of 20 objects, at 72 angles"""
    newSize = 32,32

    for obj in range(1,21):
        for angle in range(72):
            imageLocation = getLocation(obj,angle)
            im = Image.open(imageLocation)
            im.thumbnail(newSize,Image.ANTIALIAS)
            im.save("coil-20-proc/"+imageName,"PNG")

def toGrayscale(color):
    """Convert color tuple to grayscale"""
    return round(color[0]*.21+color[1]*.72+color[2]*.07)

def toArray(location):
    """Convert image to array of pixels"""
    face = Image.open(location)
    face.load()
    width, height = face.size
    rgb = face.convert("RGB")

    ar = []
    for y in range(height):
        for x in range(width):
            ar.append(toGrayscale(rgb.getpixel((x,y))))

    return ar

def getMatrix(objects,angles):
    """Return a matrix where each row is a different image's pixels"""
    imageSize = 32*32
    matrix = np.zeros((objects*angles,imageSize))
    currentRow = 0

    for obj in range(1,objects+1):
        for angle in range(angles):
            ar = toArray(getLocation(obj,angle))
            matrix[currentRow] = np.array(ar)
            currentRow+=1
    return matrix


