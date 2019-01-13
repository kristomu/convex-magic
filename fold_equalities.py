
# Given matrices A, C, and vectors b, d representing the polytope
# Ax <= b
# Cx = d, create a derived representation A'x' <= b' where any feasible
# point x' can be translated into an x so that Ax <= b, Cx = d.

# Example problem:

# A = |-1  0  0  0 | b = |0|  C = |1 1 1 0| d = | 50|
#     | 0 -1  0  0 |     |0|      |2 3 4 0|     |158|
#     | 0  0 -1  0 |     |0|      |4 3 2 0|     |142|
#     | 0  0  0 -1 |     |0|      
#     | 0  0  0  1 |     |1|

# Our strategy is to perform an LU decomposition C = PLU, then modify
# d to d' = L^-1P^-1d so as to get Ux = d'. Then we can do simultaneous
# backsubstitution on U and d' to get U' in rref, and derive F and d'' so 
# that Ax <= b and Cx = d <=> AFx' <= b - Ad'.

import numpy as np
from scipy.linalg import lu

def reduce_echelon(U_in, d_mark_in):
	U = np.copy(U_in)
	d_mark = np.copy(d_mark_in)

	# Normalize so the diagonals are one.
	diag_U = np.copy(np.diag(U))
	diag_U[diag_U == 0] = 1			# required to avoid division by zero

	U = U / diag_U[:,np.newaxis] 	# row-wise division
	d_mark = d_mark / diag_U

	# back-substitution (and counting the number of non-free variables)
	num_nonfree = 0
	for y in xrange(diag_U.shape[0]-1, 0, -1):
		coefficient_previous_row = U[y-1][y]
		coefficient_this_row = U[y][y]

		# numerical instability issues?
		if coefficient_previous_row == 0 or coefficient_this_row == 0: 
			continue
		num_nonfree += 1

		# Note: coefficient_this_row should be either 0 or 1.
		U[y-1] -= U[y] * coefficient_previous_row / coefficient_this_row
		d_mark[y-1] -= d_mark[y] * coefficient_previous_row / coefficient_this_row

	if U[0][0] != 0:
		num_nonfree += 1

	return U, d_mark, num_nonfree

def incorporate_equality(A, b, C, d):
	P, L, U = lu(C)
	# numerical instability issues?
	d_mark = np.dot(np.dot(np.linalg.inv(L), np.linalg.inv(P)), d)

	U, d_mark, num_nonfree = reduce_echelon(U, d_mark)

	# F is now a block matrix consisting of an upper (n-f)xf block giving 
	# the non-free variables in terms of the free variables, and a lower block
	# that's an fxf identity matrix (giving the free variables in terms of
	# themselves). 

	# The number of variables is the number of columns (not rows) in C, as
	# some variables may be trivially free and thus not given a row in C.
	num_free = C.shape[1] - num_nonfree

	# The upper block is negated because the row of U gives the free and
	# nonfree variables, e.g. x + y - z = 30. To get that in terms of x,
	# we need to move all the other variables to the RHS, which requires
	# a negation: x = -y + z + 30.
	upper = -U[:num_nonfree][:,-num_free:]

	F = np.array(np.bmat([[upper], [np.eye(num_free)]]))
	x_0 = np.pad(d_mark, (0, C.shape[1]-d_mark.shape[0]), "constant")
	Ax0 = np.dot(A, x_0)

	# The reduced problem is AFz <= b - Ax0.
	# To turn z to x, x = Fz + x_0.

	return np.dot(A, F), b - Ax0, F, x_0

if __name__ == "__main__":

	A = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], 
					[0,0,0,-1], [0, 0, 0, 1]])
	b = np.array([0, 0, 0, 0, 1])
	C = np.array([[1, 1, 1, 0], [2, 3, 4, 0], [4, 3, 2, 0]])
	d = np.array([50, 158, 142])

	AF, bn, F, x_0 = incorporate_equality(A, b, C, d)

	# For verification purposes, consider the linear program

	# max cTx s.t. Ax <= b, Cx=d
	# with c being [0, 1, 0, 0]
	# The optimal solution is x = [0, 42, 8, 0].

	# The modified version is
	# max cTFz s.t. AFz <= b - Ax0

	# The test consists of running an LP against both the unmodified problem
	# and the problem with equality constraints incorporated.

	from scipy.optimize import linprog

	c = np.array([0, -1, 0, 0])
	x_opt = linprog(c, A, b, C, d)["x"] # (0, 42, 8, 0)

	cF = np.dot(c, F)
	z_opt = linprog(cF, AF, bn)["x"]

	x_sec_opt = np.dot(F, z_opt) + x_0

	if np.all(x_sec_opt == x_opt):
		print "Passed test."
	else:
		print "Failed test."