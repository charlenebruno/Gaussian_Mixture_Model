#GMM algorithm

import numpy as np

import math as math

##initialize parameters teta

J = 2 #nb of clusters, arbitrary

N = 3 #nb of dimensions


phi = [1/J,1/J]

mu = np.array([[20,20],
                [10,10],
                [15,15]])

covariance_matrix = np.array(
                [[[0.7,0,0],
                [0,0.3,0],
                [0,0,0.2]],

                [[0.5, 0,0],
                [0, 0.8,0],
                 [0, 0,0.1]]])
print(covariance_matrix.shape)


n_samples = 300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered
shifted_gaussian = np.dot(np.random.randn(n_samples, N), covariance_matrix[0,:,:]) + mu[:,0]


#print(shifted_gaussian.shape)
# generate zero centered stretched Gaussian data
stretched_gaussian = np.dot(np.random.randn(n_samples, N), covariance_matrix[1,:,:]) + mu[:,1]

#print(stretched_gaussian.shape)
# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])
print(X_train.shape)

##GMM algorithm

#E-step

def e_step(X_train, phi, mu, covariance):
    I = len(X_train)
    Wj = np.zeros((I,J))

    for i in range(I):
        for j in range(J):
            sumInf = 0
            for k in range (J):
                fracInf = (1/(2*math.pi)**N/2)*np.linalg.det(covariance[k,:,:])**0.5
                soustractionInf = np.reshape((X_train[i]-mu[:,k]),[N,1])
                transposeInf = np.transpose(soustractionInf)
                invCovarianceInf = np.linalg.inv(covariance[k,:,:])

                expInf = np.exp(-0.5*(transposeInf.dot(invCovarianceInf)).dot(soustractionInf))*phi[k]

                sumInf += fracInf*expInf


            detCovariance = np.linalg.det(covariance[j,:,:])
            fracSup = 1/(2*math.pi*detCovariance**0.5)
            transposeSup = np.transpose(X_train[i] - mu[:,j])
            invCovarianceSup = np.linalg.inv(covariance[j,:,:])
            soustractionSup = np.reshape((X_train[i]-mu[:,j]),[N,1])
            expSup = np.exp(-0.5*(transposeSup.dot(invCovarianceSup)).dot(soustractionSup)*phi[j])

            Wj[i][j] = (fracSup*expSup)/sumInf

    return Wj

#M-step
def m_step(Wj,X_train):

    #initialization
    I = len(X_train)
    mu = np.zeros((N,J))
    covariance = np.zeros((J,N,N))
    phi = np.zeros([1,J])
    sumWj = np.sum(Wj) #Sum of each column of Wj

    for j in range(J):

        #we compute mu-j
        sumMu = 0
        for i in range(I):
            sumMu += Wj[i][j]*X_train[i]
        mu[:,j] = sumMu / sumWj

        #we compute phi-j
        phi[:,j] = sumWj/I

        #we compute the covariance-j
        sumCovariance = 0
        substraction = X_train[i]-mu[:,j]
        for i in range(I):
            transpose = np.transpose(substraction)
            sumCovariance += Wj[i][j]*(substraction).dot(transpose)
        covariance[j,:,:] = sumCovariance/sumWj
    return(phi, mu, covariance)


##Find optimal parameters teta*
Wj = e_step(X_train,phi,mu,covariance_matrix)
print(Wj.shape)
#print(Wj)
phi, mu, covariance = m_step(Wj, X_train)

print(covariance)
##Find cluster Ychapeau*