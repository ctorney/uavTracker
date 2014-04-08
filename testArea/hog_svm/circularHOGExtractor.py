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
    mNBins = 6
    def __init__(self, bins=3, smooth=6, modes=5):

        # number of bins in the radial direction for large scale features
        self.mNBins = bins
        # number of fourier modes that will be used (0:modes-1)
        self.mNModes = modes 
        # range of smoothing convolution in pixels
        self.mNSmooth = smooth

        # triangular kernel
        [x,y]=np.meshgrid(range(-self.mNSmooth+1,self.mNSmooth),range(-self.mNSmooth+1,self.mNSmooth))
        z = x + 1j*y
        self.trKernel = self.mNSmooth - np.abs(z)
        self.trKernel[self.trKernel < 0] = 0
        self.trKernel = self.trKernel/sum(sum(self.trKernel))

        self.ciKernel = []
        modes = range(0, order+1)
        Scale = range(1, bins+1)
        sigma = smooth;

        for s in Scale:
            r = int(smooth * s);
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

                self.ciKernel.append( kernel);



    def extract(self, img):

global chKERNEL triKERNEL triKERNEL2
%%
[om,on,~] = size(I);
padSize = (self.mNSmooth*5);
%% get gradient
% if not grayscale then combine each colour channel
if(size(I,3) == 3)
    [DX,DY,~] = gradient(I);
    mag = DX .^ 2 + DY .^ 2;
    [~,channel] = max(mag,[],3);
    dx = DX(:,:, 1) .* (channel == 1) + DX(:,:, 2) .* (channel == 2) + DX(:,:, 3) .* (channel == 3);
    dy = DY(:,:, 1) .* (channel == 1) + DY(:,:, 2) .* (channel == 2) + DY(:,:, 3) .* (channel == 3);
    complex_g = complex(dx,dy);
else
    [dx,dy] = gradient(I);
    complex_g = complex(dx,dy);
end

complex_g = padarray(complex_g,[padSize,padSize],0);
%% project to fourier space
[m,n] = size(complex_g);
order = [0 1 2 3 4];  % only contrast-insensitive? and postive frequncies
f_g = zeros([m,n,numel(order)]);
phase_g = angle(complex_g);
mag_g = abs(complex_g);

% local gradient magnitude normalization, scale = sigma * 2
local_mag_g = sqrt(conv2(mag_g.^2, triKERNEL2, 'same'));
mag_g = mag_g ./ (local_mag_g + 0.001);

for j = 1:numel(order)
    f_g(:,:,j) = exp( -1i * (order(j)) * phase_g) .* mag_g;
end
f_g(:,:,1) = f_g(:,:,1) * 0.5;

local_mag_g = unPad(local_mag_g, [om,on]);
%% compute regional description by convolutions
center_f_g = zeros(om,on,numel(order));
template = triKERNEL;
c_featureDetail = [];
for j = 1:numel(order)
% 2d convolution to smooth data according to a triangle
    center_f_g(:,:,j) = unPad(conv2(f_g(:,:,j), template, 'valid'), [om,on]);
    c_featureDetail = [c_featureDetail; [0,-1,-order(j), 0, -order(j)]];
end

% frequncy control
maxFreq = 4;
maxFreqSum = 4;

% count output feature channels
nScale = size(chKERNEL,1);
featureDetail = [];
for s = 1:nScale
    featureDim = 0;
    for freq = -maxFreq:maxFreq
        for j = 1:numel(order)
            ff = -(order(j))+freq;
            if(ff >= -maxFreqSum && ff <= maxFreqSum && ~(order(j)==0 && freq < 0))
                featureDim = featureDim + 1;
                featureDetail = [featureDetail; [s,-1,-order(j), freq, ff]];
            end
        end
    end
end

% compute convolutions
fHoG = zeros([om,on,featureDim*nScale]);
cnt = 0;
for s = 1:nScale
    for freq = -maxFreq:maxFreq
        template = chKERNEL{s,abs(freq)+1};
        if(freq < 0)
            template = conj(template);
        end
        for j = 1:numel(order)
            ff = -(order(j))+freq;
            if(ff >= -maxFreqSum && ff <= maxFreqSum && ~(order(j)==0 && freq < 0))
                cnt = cnt + 1;
                fHoG(:,:,cnt) = unPad(conv2(f_g(:,:,j), template, 'valid'), [om,on]);
            end
        end
    end
end

%% visualization

% for s = 1:nScale
%     figure('Name',['Features at scale ' , num2str(s)]);
%     set(gcf, 'Color',[1,1,1]);
%     n = ceil(sqrt(featureDim));
%     HH = tight_subplot(n,n,0.02,0.05,0.05);
%
%     for i = 1:featureDim
%         axes(HH(i));imagesc(real(fHoG(:,:,(s-1) * featureDim + i)));axis equal tight off;colormap jet
%     end
% end

%%
fHoG = reshape(fHoG, om*on, size(fHoG,3));

center_f_g = reshape(center_f_g, om*on, size(center_f_g,3));
%% create invariant features
% some features are naturally rotation-invariant
iF_index = featureDetail(:,end) == 0;
iF = fHoG(:, iF_index);
% for complex number
ifreal = false(1, size(iF,2));
for i = 1:size(iF,2)
    ifreal(i) = isreal(iF(:,i));
end
iF = [real(iF), imag(iF(:,~ifreal))];
i_featureDetail = featureDetail(iF_index,:);
i_featureDetail = [i_featureDetail; i_featureDetail(~ifreal,:)];


% generate magnitude features from the non-invariant features
mF = abs([fHoG(:,~iF_index) center_f_g local_mag_g(:)]);
mFdetail = [featureDetail(~iF_index,:) ; c_featureDetail; [0, -1, -1,-1,-1]];


% coupling features across different radius/scales
% here we couple radius 1/2 and radius 2/3
cF = cell(1,2);
cFdetail = cell(2,1);
for i = 1:2
    j = i+1;
    cF{i} = fHoG(:, (i-1) * featureDim + (1:featureDim) ) .* conj( fHoG(:, (j-1) * featureDim + (1:featureDim) )  );
    cF{i} = cF{i} ./ (sqrt(abs(cF{i})) + eps);      % take magnitude but keep the phase
    cFdetail{i} = featureDetail((i-1) * featureDim + (1:featureDim),:);
    cFdetail{i}(:,2) = featureDetail((j-1) * featureDim + (1:featureDim),1);
end
cF = cell2mat(cF);
cFdetail = cell2mat(cFdetail);
% for complex number
ifreal = false(1, size(cF,2));
for i = 1:size(cF,2)
    ifreal(i) = isreal(cF(:,i));
end
cF = [real(cF), imag(cF(:,~ifreal))];
cFdetail =[cFdetail; cFdetail(~ifreal,:)];

%% final output
Feature = [iF mF cF];
Fdetail = [i_featureDetail; mFdetail; cFdetail];

end
        
	"""
        Extract the line orientation and and length histogram.
        """
        #I am not sure this is the best normalization constant. 
        retVal = []
        p = max(img.width,img.height)/2
        minLine = 0.01*p
        gap = 0.1*p
        fs = img.findLines(threshold=10,minlinelength=minLine,maxlinegap=gap)
        ls = fs.length()/p #normalize to image length
        angs = fs.angle()
        lhist = np.histogram(ls,self.mNBins,normed=True,range=(0,1))
        ahist = np.histogram(angs,self.mNBins,normed=True,range=(-180,180))
        retVal.extend(lhist[0].tolist())
        retVal.extend(ahist[0].tolist())
        return retVal


    
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

