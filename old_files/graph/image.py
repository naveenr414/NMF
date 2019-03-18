from PIL import Image
import numpy as np

def extractFaces(location):
    """Extract the faces from the 19x19 image file"""
    faceSize = 19
    
    allFaces = Image.open(location)
    allFaces.load()
    width, height = allFaces.size

    rgb = allFaces.convert("RGB")

    imageNumber = 1
    for xStart in range(0,width,faceSize+2):
        for yStart in range(0,height,faceSize+2):
            data = np.zeros((faceSize,faceSize,3),dtype=np.uint8)

            for x in range(faceSize):
                for y in range(faceSize):
                    data[y][x] = rgb.getpixel((x+xStart,y+yStart))

            img = Image.fromarray(data,'RGB')
            img.save("faces/"+str(imageNumber)+".png")
            imageNumber+=1

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

def writeGrayscale(ar,name):
    """Write the grayscale array to file with filename 'name'"""
    size = round(len(ar)**.5)
    newAr = np.zeros((size,size,3),dtype=np.uint8)
    
    for i in range(0,len(ar),size):
        for j in range(size):
            newAr[i//size][j] = (ar[i+j],ar[i+j],ar[i+j])

   
    img = Image.fromarray(newAr,'RGB')
    img.save(name)

writeGrayscale(toArray("faces/1.png"),"faces/gray_1.png")   
