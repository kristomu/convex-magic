#pragma once
#include "../polytope.h"
#include <eigen3/Eigen/Dense>

// Approximate the maximum-distance l2 diameter of a convex polytope by
// determining the maximum-distance l1 diameter, i.e.

// max ||x-y||_1 subject to Ax <= b, Ay <= b.

// This is a hard problem and thus requires mixed integer programming.
// In the worst case, determining the diameter will take far too long
// a time, but we'll deal with that when it happens.

// The maximum diameter is used to improve mixing times for the billiard
// walk.

// TODO: Fix redundancy (get_simplex and linear_program).

typedef std::pair<Eigen::VectorXd, Eigen::VectorXd> diameter_coords;

class polytope_distance {

	public:
		diameter_coords get_extreme_coords(const polytope & poly_in, 
			double M) const;

		double get_l1_diameter(const polytope & poly_in, double M) const;

		double get_l2_diameter_lb(const polytope & poly_in, 
			double M) const;

		// Using bounding boxes to infer M.

		double get_l1_diameter(const polytope & poly_in) const;
		double get_l2_diameter_lb(const polytope & poly_in) const;
};