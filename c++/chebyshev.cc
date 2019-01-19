// Determining the Chebyshev center of a polytope.
// TODO: Fix duplicate code. Also explain how the calculation works.

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <glpk.h>

#include <random>
#include <numeric>
#include <stdexcept>

#include "polytope.h"

polytope get_simplex(int dimension) {

	polytope out;
	out.A = Eigen::MatrixXd(dimension+1, dimension);
	out.b = Eigen::VectorXd(dimension+1);
	int i;

	for (i = 0; i < dimension; ++i) { out.b[i] = 0; }
	out.b[dimension] = 1;

	for (i = 0; i < dimension; ++i) {
		out.A(i, i) = -1;
		out.A(dimension, i) = 1;
	}

	return out;
}

std::pair<double, Eigen::VectorXd> linear_program(
	const Eigen::MatrixXd & A, const Eigen::VectorXd & b, 
	const Eigen::VectorXd & c) {

	size_t A_size = A.rows() * A.cols();

	// A in sparse representation
	int row_idx[A_size], col_idx[A_size];
	double value[A_size];

	glp_prob *lp;
	lp = glp_create_prob();
	glp_set_prob_name(lp, "eigen_lp");
	glp_set_obj_dir(lp, GLP_MIN);
	glp_add_rows(lp, A.rows());
	glp_term_out(GLP_OFF);			// suppress output 
	size_t i;

	// set b
	char rowname[] = "b_0";
	for (i = 0; i < A.rows(); ++i) {
		rowname[2] = i + '0';

		glp_set_row_name(lp, i+1, rowname);
  		glp_set_row_bnds(lp, i+1, GLP_UP, 0.0, b(i));
	}

	// set x and c
	char colname[] = "x_0";
	glp_add_cols(lp, A.cols());
	for (i = 0; i < A.cols(); ++i) {
		colname[2] = i + '0';
		glp_set_col_name(lp, i+1, colname);
		glp_set_col_bnds(lp, i+1, GLP_FR, 0.0, 0.0);
		glp_set_obj_coef(lp, i+1, c(i));
	}

	// set A
	int row, col;
	i = 0;
	for (row = 0; row < A.rows(); ++row) {
		for (col = 0; col < A.cols(); ++col) {
			if (A(row, col) == 0) {
				continue;
			}
			row_idx[1+i] = row+1;
			col_idx[1+i] = col+1;
			value[1+i] = A(row, col);
			++i;
		}
	}

	// load and solve
	glp_load_matrix(lp, i, row_idx, col_idx, value);
	glp_simplex(lp, NULL); // solve the LP

	// return value at optimal point, and that point.
	Eigen::VectorXd x_opt(A.cols());

	for (i = 0; i < A.cols(); ++i) {
		x_opt(i) = glp_get_col_prim(lp, i+1);  		
	}

	double z = glp_mip_obj_val(lp);

	return std::pair<double, Eigen::VectorXd>(z, x_opt);
}

// Augment the A matrix of a polytope, producing A_*, so that the linear
// program max x s.th. A_*x <= b provides the radius of the maximum-radius
// ball that can be contained inside the polytope Ax <= b, and the
// optimal point consists of the coordinates of the center of that ball 
// (as well as its radius r).

Eigen::MatrixXd chebyshev_augment(const Eigen::MatrixXd & A) {

	// Get each polytope row's l2 norm into a vector.

	Eigen::VectorXd radius_weight = A.rowwise().norm();
	
	Eigen::MatrixXd augmented_A(A.rows(), A.cols()+1);
	// Add A
	augmented_A.block(0, 0, A.rows(), A.cols()) = A;
	// Set the additional (rightmost) column to the r_weights vector.
	augmented_A.block(0, augmented_A.cols()-1, augmented_A.rows(), 1) =\
		radius_weight;

	return augmented_A;
}

std::pair<double, Eigen::VectorXd> get_chebyshev_center_w_radius(
	const polytope & poly_in) {

	// maximize r, i.e. minimize 0 0 0 0 ... -1, since r is the last
	// variable.

	Eigen::VectorXd c = Eigen::VectorXd::Zero(poly_in.A.cols()+1);
	c[poly_in.A.cols()] = -1;
	
	std::pair<double, Eigen::VectorXd> out =
		linear_program(chebyshev_augment(poly_in.A), poly_in.b, c);

	// Strip r away from the output vector because we only want it to
	// contain the coordinates themselves.

	out.second.conservativeResize(out.second.size()-1);

	return out;
}

Eigen::VectorXd get_chebyshev_center(const polytope & poly_in) {
	return get_chebyshev_center_w_radius(poly_in).second;
}

main() {
	int dimension = 2;
	polytope simplex = get_simplex(dimension);

	Eigen::VectorXd initial_point = get_chebyshev_center(simplex);
	Eigen::Vector2d should_be;
	should_be << 1 / (2.0 + sqrt(2)), 1 / (2.0 + sqrt(2));

	if (should_be == initial_point) {
		std::cout << "Test PASS" << std::endl;
	} else {
		std::cout << "Test FAIL" << std::endl;
	}
}