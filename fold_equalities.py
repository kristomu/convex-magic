
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

import numpy as np
from scipy.linalg import null_space, lstsq

def incorporate_equality(A, b, C, d):
	# Now use the following strategy: Any solution to Cx=d is given by
	# Fz + x_0, where x_0 is some solution to Cx=d, and F is the null
	# space of C.
	# Much easier than mucking about with backsubstitution.

	F = null_space(C)

	# This makes it easier to read the transformed matrices.
	F = F / F.max(axis = 0)

	# Any solution to Cx=d will do, but this is the only that scipy
	# natively supports.
	x_0 = lstsq(C, d)[0]

	# The reduced problem is AFz <= b - Ax0.
	# To turn z to x, x = Fz + x_0.
	Ax0 = np.dot(A, x_0)
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

	# We're no longer constrained to nonnegative solutions, so explicitly
	# state that we're not because otherwise the test fails.
	cF = np.dot(c, F)
	z_opt = linprog(cF, AF, bn, bounds=((None, None), (None, None)))["x"]

	x_sec_opt = np.dot(F, z_opt) + x_0

	if np.all(np.abs(x_sec_opt - x_opt) < 1e-12):
		print "Passed test."
	else:
		print "Failed test."