import numpy as np
def testCirc(binSize, order):

    retVal = []
    bins = 3;
    modes = np.linspace(0, order, order+1)
    Scale = np.linspace(1, bins, bins)
    sigma = binSize;

    for s in Scale:
        r = int(binSize * s);
        ll = range(1-r-sigma,r+sigma)
        [x,y] = np.meshgrid(ll,ll)
        z = x + 1j*y
        phase_z = np.angle(z);
                            
        for j in modes:
            kernel = sigma - np.abs(np.abs(z) - r) 
            kernel[kernel < 0] = 0
            kernel = np.multiply(kernel,np.exp(1j*phase_z*j))
            sa = np.ravel(np.abs(kernel))
            kernel = kernel / np.sqrt(np.sum(np.multiply(sa,sa)))
            #kernel = kernel / sqrt(sum(abs(kernel(:)).^2));     %% make all the output value in same range
            retVal.append( kernel);



    return retVal
