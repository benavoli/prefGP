#
# Slice sampling
#
import numpy as np
from tqdm import tqdm


class ess:
    def __init__(self,initial_theta,L,lnpdf,pdf_params=(),scalingf=[]):
        """
        INPUT:
        lnpdf - function evaluating the log of the pdf to be sampled
        pdf_params - parameters to pass to the pdf
        scalingf - returnes a matrix (e.g., Cholesky) that multiples the samples from N(0,I) 
        """
        self.initial_theta = initial_theta
        self.lnpdf=lnpdf
        self.pdf_params=pdf_params
        self.scalingf = scalingf
        self.L=L
        
    def sample(self,nsamples,tune=50,progress=True):
        cur_lnpdf = None
        xx_prop = self.initial_theta
        pbar = tqdm(total = nsamples+tune, disable=1-progress, position=0, leave=True)
        SS = []
        for i in range(nsamples+tune):
            (xx_prop,cur_lnpdf)=ess_step(xx_prop,self.L,self.lnpdf,pdf_params=(),
                     cur_lnpdf=cur_lnpdf,angle_range=None)
            SS.append(xx_prop)
            pbar.update(1)
        pbar.close()
        return np.vstack(SS)[tune:,:]
        
    
def ess_step(initial_theta,prior,lnpdf,pdf_params=(),
                     cur_lnpdf=None,angle_range=None):
    """
    INPUT:
       initial_theta - initial vector
       L - cholesky decomposition of the covariance matrix 
               (like what np.linalg.cholesky returns), 
               or a sample from the prior
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params - parameters to pass to the pdf
       cur_lnpdf - value of lnpdf at initial_theta (optional)
       angle_range - Default 0: explore whole ellipse with break point at
                    first rejection. Set in (0,2*pi] to explore a bracket of
                    the specified width centred uniformly at random.
    OUTPUT:
       new_theta, new_lnpdf
    REFERENCE:
    Murray, Iain, Ryan Adams, and David MacKay. "Elliptical slice sampling." AISTAT 2010.

    """
    D= len(initial_theta)
    if cur_lnpdf is None:
        cur_lnpdf= lnpdf(initial_theta,*pdf_params)

    # Set up the ellipse and the slice threshold
    if len(prior.shape) == 1: #prior = prior sample
        nu= prior
    else: #prior = cholesky decomp
        if not prior.shape[0] == D or not prior.shape[1] == D:
            raise IOError("Prior must be given by a D-element sample or DxD chol(Sigma)")
        nu= prior@np.random.normal(size=D)
    hh = np.log(np.random.uniform()) + cur_lnpdf

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range is None or angle_range == 0.:
        # Bracket whole ellipse with both edges at first proposed point
        phi= np.random.uniform()*2.*np.pi
        phi_min= phi-2.*np.pi
        phi_max= phi
    else:
        # Randomly center bracket on current point
        phi_min= -angle_range*np.random.uniform()
        phi_max= phi_min + angle_range
        phi= np.random.uniform()*(phi_max-phi_min)+phi_min

    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference and check if it's on the slice
        
        xx_prop = initial_theta*np.cos(phi) + nu*np.sin(phi)
        cur_lnpdf = lnpdf(xx_prop,*pdf_params)
        
        if cur_lnpdf > hh:
            # New point is on slice, ** EXIT LOOP **
            break
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')
        # Propose new angle difference
        phi = np.random.uniform()*(phi_max - phi_min) + phi_min
    return (xx_prop,cur_lnpdf)
    
def liness_step(x00, AA, bb, mean, C, nsamples,  tune=100, progress=True):
    """
    Sample from a Gaussian vector x \sim N(mean,C) in the region AA@x>=bb.

    mean is a vector indicating the mean of the Gaussian distribution

    C is a matrix indicating the covariance of the gaussianCDF

    nsamples is an integer that indicates the number of samples

    tune is an integer indicating the burn-in samples for the methods

    """
    import scipy as sp
    try:
        eigen_max = sp.linalg.eigh(C,eigvals_only=True,eigvals=(C.shape[0]-1,C.shape[0]-1))
    except:
        eigen_max = np.max(np.trace(C))
    #eigen_max = 1.0
    #print(eigen_max)
    def normalize(a,b):
        if (a > b) and not (a<0 and b<0):
            a -= 2*np.pi
        elif (a > b) and (a<0 and b<0):
            #a1= a+2*np.pi
            b = 2*np.pi+b
        return a,b
    def getOverlap(a, b):
            return np.hstack([max(a[0], b[0]),min(a[1], b[1])])
    def intersection(a,b,c,d):
        A, B = normalize(a, b)
        C, D = normalize(c, d)
        if B<C or D<A:
            raise("empty intersection")
        I_1 = getOverlap([A, B], [C, D])
        return I_1
    b=bb/np.sqrt(eigen_max)
    x00 = x00/np.sqrt(eigen_max)
    #x0 = (b + 0.1*np.random.rand(len(b),1))
    Q=[]
    L=np.linalg.cholesky(C/eigen_max)
    mc = 0
    pbar = tqdm(total = nsamples+tune, disable=1-progress, position=0, leave=True)
    while mc < nsamples+tune:
        nu0 = L@np.random.randn(L.shape[0],1)
        nu = AA@(nu0)
        #print(nu0[0:5])
        x0 = AA@x00
        r_sq = (x0)**2 +(nu)**2
        

        thetas_1=2*np.arctan2(nu-np.sqrt(r_sq -b**2),x0+b)#+np.pi
        thetas_2=2*np.arctan2(nu+np.sqrt(r_sq -b**2),x0+b)#+np.pi
        I1= [-2*np.pi,2*np.pi]
        vmin=-2*np.pi
        vmax= 2*np.pi
        empty = True

        for i in range(thetas_1.shape[0]):
            if np.isnan(thetas_1[i,0]+thetas_2[i,0])==0:
                empty = False
                eps=np.mod(abs(thetas_2[i,0]-thetas_1[i,0]),2*np.pi)*0.001
                #print([thetas_1[i],thetas_2[i]])
                if x0[i]*np.cos(thetas_1[i]+eps) +nu[i]*np.sin(thetas_1[i]+eps)>=b[i]:
                    vmin = thetas_1[i]
                    vmax = thetas_2[i]
                elif x0[i]*np.cos(thetas_1[i]-eps) +nu[i]*np.sin(thetas_1[i]-eps)>=b[i]:
                    vmin = thetas_2[i]
                    vmax = thetas_1[i]
                I1=intersection(np.copy(I1[0]),np.copy(I1[1]),vmin,vmax)
        if empty == False:
            theta=I1[0]+np.random.rand(1)*np.mod((I1[1]-I1[0]),2*np.pi)
            #print(theta)
            x_sample = x00*np.cos(theta)+nu0*np.sin(theta)

            if np.min(-b+AA@x_sample)<0:
                raise("error: the constraint is violated")
            Q.append(x_sample[:,0]*np.sqrt(eigen_max)+mean)
            x00 = x_sample
            mc=mc+1
        else:
            theta=np.random.rand(1)*2*np.pi
            x_sample = x00*np.cos(theta)+nu0*np.sin(theta)
            if np.min(-b+AA@x_sample)<0:
                raise("error: the constraint is violated")
            Q.append(x_sample[:,0]*np.sqrt(eigen_max)+mean)
            x00 = x_sample
            mc=mc+1
        pbar.update(1)
    pbar.close()
    res= np.array(Q)[tune:,:]
    return res


