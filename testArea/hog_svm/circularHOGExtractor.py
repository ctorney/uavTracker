from SimpleCV.base import *
from SimpleCV.ImageClass import Image
from SimpleCV.Features.FeatureExtractorBase import *


class circularHOGExtractor(FeatureExtractorBase):
    """
    Create a 1D edge length histogram and 1D edge angle histogram.
    
    This method takes in an image, applies an edge detector, and calculates
    the length and direction of lines in the image.
    
    bins = the number of bins
    """
    def __init__(self, bins=4, size=6, max_freq=4):

        # number of bins in the radial direction for large scale features
        self.mNBins = bins
        # size of bin in pixels, this sets the required radius for the image = bins*size
        self.mNSize = size
        # number of fourier modes that will be used (0:modes-1)
        self.mNMaxFreq = max_freq 

        # create a list to store kernels for regional descriptors based on circular harmonics
        self.ciKernel = []

        # first create the central region 
        [x,y]=np.meshgrid(range(-self.mNSize+1,self.mNSize),range(-self.mNSize+1,self.mNSize))
        z = x + 1j*y
        kernel = self.mNSize - np.abs(z)
        kernel[kernel < 0] = 0
        kernel = kernel/sum(sum(kernel))

        self.ciKernel.append(kernel)

        # next build the internal regions - (bins-1) concentric circles
        modes = range(0, self.mNMaxFreq+1)
        Scale = range(2, bins+1)

        for s in Scale:
            r = int(self.mNSize * s)
            ll = range(1-r,r)
            [x,y] = np.meshgrid(ll,ll)
            z = x + 1j*y
            phase_z = np.angle(z);
                            
            for j in modes:
                kernel = self.mNSize - np.abs(np.abs(z) - r) 
                kernel[kernel < 0] = 0
                kernel = np.multiply(kernel,np.exp(1j*phase_z*j))
                sa = np.ravel(np.abs(kernel))
                kernel = kernel / np.sqrt(np.sum(np.multiply(sa,sa)))

                self.ciKernel.append(kernel)



    def extract(self, img):
        I = img.getGrayNumpyCv2().astype(float)/255.0
        (nx, ny) = I.shape
        cx = int(round(0.5*nx))
        cy = int(round(0.5*ny))
        (dy, dx) = np.gradient(I)
        dz = dx + 1j*dy

        phi = np.angle(dz)
        r = np.abs(dz)
        histF = np.zeros([nx,ny, self.mNMaxFreq+1])+0j

        for k in range(0,self.mNMaxFreq+1):
            histF[:,:,k] = np.multiply(np.exp( -1j * (k) * phi) , r+0j)
        
        histF[:,:,0] = histF[:,:,0] * 0.5;
        return I


    
    def getFieldNames(self):
        """
        Return the names of all of the length and angle fields. 
        """
        retVal = []
        for i in range(self.mNBins):
            name = "Length"+str(i)
            retVal.append(name)
        for i in range(self.mNBins):
            name = "Angle"+str(i)
            retVal.append(name)
                        
        return retVal
        """
        This method gives the names of each field in the feature vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """

    def getNumFields(self):
        """
        This method returns the total number of fields in the feature vector.
        """
        return self.mNBins*2

