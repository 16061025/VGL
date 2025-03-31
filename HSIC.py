"""
python implementation of Hilbert Schmidt Independence Criterion
hsic_gam implements the HSIC test using a Gamma approximation
Python 2.7.12

Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., 
& Smola, A. J. (2007). A kernel statistical test of independence. 
In Advances in neural information processing systems (pp. 585-592).

Shoubo (shoubo.sub AT gmail.com)
09/11/2016

Inputs:
X 		n by dim_x matrix
Y 		n by dim_y matrix
alph 		level of test

Outputs:
testStat	test statistics
thresh		test threshold for level alpha test
"""

from __future__ import division
import numpy as np
import torch
from scipy.stats import gamma

def rbf_dot(pattern1, pattern2, deg):
	size1 = pattern1.shape
	size2 = pattern2.shape

	G = torch.sum(pattern1*pattern1, 1).reshape(size1[0],1)
	H = torch.sum(pattern2*pattern2, 1).reshape(size2[0],1)

	Q = torch.tile(G, (1, size2[0]))
	R = torch.tile(H.T, (size1[0], 1))

	H = Q + R - 2* torch.mm(pattern1, pattern2.T)

	H = torch.exp(-H/2/(deg**2))

	return H


def hsic_corr(X, alph = 0.5):
	hsic_distance_matrix = torch.eye(len(X))
	for i in range(len(X)):
		for j in range(i+1,len(X)):
			corr_ij = hsic_gam_tensor(X[i], X[j], alph)
			hsic_distance_matrix[i][j] = corr_ij
			hsic_distance_matrix[j][i] = corr_ij
	return hsic_distance_matrix


def hsic_gam_tensor(X, Y, alph = 0.5):
	"""
	X, Y are tensor vectors with row - sample, col - dim
	alph is the significance level
	auto choose median to be the kernel width
	"""
	n = X.shape[0]

	# ----- width of X -----
	Xmed = X

	G = torch.sum(Xmed*Xmed, 1).reshape(n,1)
	Q = torch.tile(G, (1, n) )
	R = torch.tile(G.T, (n, 1) )

	dists = Q + R - 2* torch.mm(Xmed, Xmed.T)
	dists = dists - torch.tril(dists)
	dists = dists.reshape(n**2, 1)

	width_x = torch.sqrt( 0.5 * torch.median(dists[dists>0]) )
	# ----- -----

	# ----- width of X -----
	Ymed = Y

	G = torch.sum(Ymed*Ymed, 1).reshape(n,1)
	Q = torch.tile(G, (1, n) )
	R = torch.tile(G.T, (n, 1) )

	dists = Q + R - 2* torch.mm(Ymed, Ymed.T)
	dists = dists - torch.tril(dists)
	dists = dists.reshape(n**2, 1)

	width_y = torch.sqrt( 0.5 * torch.median(dists[dists>0]) )
	# ----- -----

	bone = torch.ones((n, 1))
	H = torch.eye(n) - torch.ones((n,n)) / n

	K = rbf_dot(X, X, width_x)
	L = rbf_dot(Y, Y, width_y)

	Kc = torch.mm(torch.mm(H, K), H)
	Lc = torch.mm(torch.mm(H, L), H)

	testStat = torch.sum(Kc.T * Lc) / n

	varHSIC = (Kc * Lc / 6)**2

	varHSIC = ( torch.sum(varHSIC) - torch.trace(varHSIC) ) / n / (n-1)

	varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

	K = K - torch.diag(torch.diag(K))
	L = L - torch.diag(torch.diag(L))

	muX = torch.mm(torch.mm(bone.T, K), bone) / n / (n-1)
	muY = torch.mm(torch.mm(bone.T, L), bone) / n / (n-1)

	mHSIC = (1 + muX * muY - muX - muY) / n

	al = mHSIC**2 / varHSIC
	bet = varHSIC*n / mHSIC

	thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]

	return testStat - thresh