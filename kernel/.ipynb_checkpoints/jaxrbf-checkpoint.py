import jax
#from jax.scipy.linalg import block_diag
def squared_distance(x,y):
    return jax.numpy.sum((x - y) ** 2)


def distmat(func, x,  y) :
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)


def pdist_squareform(x,y):
    return distmat(squared_distance, x, y)

def RBF(X1,X2,params,ARD=True):
    if ARD==True:
        lengthscale = params[0:X1.shape[1]]
        variance    = params[X1.shape[1]]    
    else:
        lengthscale = params[0:1]
        variance    = params[1]
    x = X1[..., :] / lengthscale
    y = X2[..., :] / lengthscale
    return variance * jax.numpy.exp(-0.5 * pdist_squareform(x, y))

