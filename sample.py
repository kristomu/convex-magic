# An implementation of the billiard walk algorithm given in

# GRYAZINA, Elena; POLYAK, Boris. Random sampling: Billiard walk algorithm.
# European Journal of Operational Research, 2014, 238.2: 497-504.

# as applied to an n-dimensional simplex.

# Simplex: Ax <= b. All x >= 0 and sum of all x <= 1.
# A = [ -I          ]  b = [0]
#     [-------------]      [-]
#     [ 1 1 1 ... 1 ]      [1]

import numpy as np

# The import below is only needed for the test instance. Comment it out if 
# you're going to use the functions below as a library

import matplotlib.pyplot as plt

def get_simplex(dimension):
	sum_constraint = np.array([1] * dimension)
	negident = -np.eye(dimension)

	A = np.concatenate((negident, np.array([sum_constraint])), axis=0)
	b = np.concatenate(([0]*dimension, [1]))

	return A, b

def is_inside(A, b, x):
		return np.all(np.dot(A, x) <= b)

# Note, can be larger than the dimensions!

def random_unit_vector(dimension):
	# Create a ray pointing in a random direction with unit magnitude.
	# Having unit magnitude makes k the distance to the closest edge.
	x_v = np.random.multivariate_normal([0]*dimension, np.eye(dimension))
	x_v /= np.linalg.norm(x_v)

	return x_v

def get_closest_half_plane_dist(A, b, x_p, x_v):

	# Find closest half-plane

	k_record = np.inf
	halfplane = -1

	# Note, can be larger than the number of dimensions!
	num_halfplanes = A.shape[0]

	# Hack to avoid numerical precision issues. In essence, this
	# factor gives each edge a thickness, where we'll never go from
	# inside the thickness of the edge to some other point inside
	# that band.
	k_epsilon = 1e-9

	for i in xrange(num_halfplanes):
		travel_magnitude = np.dot(x_v, A[i])

		# If x_v is parallel to the edge, skip.
		if travel_magnitude == 0:
			continue

		k = (b[i] - np.dot(x_p, A[i])) / travel_magnitude

		if k < k_record and k > k_epsilon:
			k_record = k
			halfplane = i

		# If this is true, we're at an edge. x_v may be pointing away from
		# the polytope or not. If x_v is pointing out of the polytope, then
		# the derivative of (x_p + k * x_v) * a[i] - b[i] is positive. In that
		# case, we can't travel any amount along that direction, so we need to
		# reflect.
		# Otherwise, we're still good to go.

		# Travel_magnitude is exactly this derivative, which gives us...

		if k >= 0 and k <= k_epsilon and travel_magnitude > 0:
			return (0, i, False)

	# TODO: Somehow handle if k_record is infinity and halfplane is -1.
	# This pretty much means the space is unbounded and x_v is pointing
	# in the direction that the polytope is unbounded.

	return (k_record, halfplane, True)

def billiard_walk_int(A, b, x_p, x_v, max_distance, max_reflections):

	#print "X_p: ", x_p
	#print "X_v: ", x_v
	#print "max distance: ", max_distance

	distance_remaining = max_distance

	for i in xrange(max_reflections):
		distance, halfplane, dest_inside = get_closest_half_plane_dist(
			A, b, x_p, x_v)

		# If dest_inside is true, then x_p + distance * x_v keeps us
		# inside the polytope. That means we can travel towards the
		# closest edge along the direction given by x_v. We will then
		# either exhaust our max distance, or end up at an edge with x_v
		# pointing out of the polytope.

		# On the other hand, if it's false, then we're at an edge with x_v
		# pointing outwards, so we can't travel at all before we need to
		# reflect.

		if dest_inside:
			if distance >= distance_remaining:
				#print "Exit in", i
				return x_p + x_v * distance_remaining

			distance_remaining -= distance
			x_p = x_p + x_v * distance

		# We're at an edge and need to reflect
		# TODO if we want to optimize: cache np.linalg.norm(A[halfplane])
		# or normalize each row of A and scale b appropriately.
		int_normal = -A[halfplane] / np.linalg.norm(A[halfplane])
		x_v -= 2 * np.dot(x_v, int_normal) * int_normal

	# Passed max_reflections without covering the required distance.
	return None

def billiard_walk(A, b, x_0, tau_distance, max_reflections, 
	max_retries=100):

	dimension = A.shape[1]

	candidate = None

	for i in xrange(max_retries):
		x_p = np.copy(x_0)
		x_v = random_unit_vector(dimension)

		max_distance = -tau_distance * np.log(np.random.uniform())

		candidate = billiard_walk_int(A, b, x_p, x_v, max_distance, max_reflections)

		if candidate is not None:
			return candidate

if __name__ == "__main__":

	g_dimension = 2

	# The diameter of the polytope is equal to the max distance between points
	# in it.
	max_diameter = np.sqrt(2)
	g_x_0 = np.array([0]*g_dimension)


	g_A, g_b = get_simplex(g_dimension)

	x = []
	y = []

	for i in xrange(1000):
		print i
		if not is_inside(g_A, g_b, g_x_0):
			print "!",
			continue
		g_x_0 = billiard_walk(g_A, g_b, g_x_0, max_diameter, 10 * g_dimension)
		#print g_x_0
		x.append(g_x_0[0])
		y.append(g_x_0[1])
		#print ""

	plt.scatter(x, y, s=[10])
	plt.show()