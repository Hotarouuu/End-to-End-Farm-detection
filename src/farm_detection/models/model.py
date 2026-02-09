from sklearn.naive_bayes import GaussianNB

def GNB(priors=None, var_smoothing=1e-9):
    return GaussianNB(priors=priors, var_smoothing=var_smoothing)