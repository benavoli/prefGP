import jax


def Linear(X1,X2,params,ARD=True):
    #jax.debug.print("{params}", params=params)
    variance = params[0:X1.shape[1]]
    #jax.debug.print("{variance}", variance=variance)
    x = X1 * jax.numpy.sqrt(variance)
    y = X2 * jax.numpy.sqrt(variance)
    return (x@y.T)

