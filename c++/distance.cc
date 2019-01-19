// Approximate the maximum-distance l2 diameter of a convex polytope by
// determining the maximum-distance l1 diameter, i.e.

// max ||x-y||_1 subject to Ax <= b, Ay <= b.

// This is a hard problem and thus requires mixed integer programming.
// In the worst case, determining the diameter will take far too long
// a time, but we'll deal with that when it happens.

// The maximum diameter is used to improve mixing times for the billiard
// walk.

// TODO: Fix redundancy (get_simplex and linear_program).

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

std::pair<double, Eigen::VectorXd> mixed_program(
	const Eigen::MatrixXd & A, const Eigen::VectorXd & b, 
	const Eigen::VectorXd & c, const std::vector<bool> & is_binary,
	bool verbose) {

	bool needs_integer_step = false;

	size_t A_size = A.rows() * A.cols();

	// A in sparse representation
	int row_idx[A_size], col_idx[A_size];
	double value[A_size];

	glp_prob *mip;
	mip = glp_create_prob();
	glp_set_prob_name(mip, "eigen_mip");
	glp_set_obj_dir(mip, GLP_MIN);
	glp_add_rows(mip, A.rows());
	if (!verbose) {
		glp_term_out(GLP_OFF);			// suppress output 
	}
	size_t i;

	// set b
	char rowname[] = "b_0";
	for (i = 0; i < A.rows(); ++i) {
		rowname[2] = i + '0';

		glp_set_row_name(mip, i+1, rowname);
  		glp_set_row_bnds(mip, i+1, GLP_UP, 0.0, b(i));
	}

	// set x and c
	char colname[] = "x_0";
	glp_add_cols(mip, A.cols());
	for (i = 0; i < A.cols(); ++i) {
		colname[2] = i + '0';
		glp_set_col_name(mip, i+1, colname);
		glp_set_col_bnds(mip, i+1, GLP_FR, 0.0, 0.0);
		glp_set_obj_coef(mip, i+1, c(i));

		if (is_binary[i]) {
			needs_integer_step = true;
			glp_set_col_kind(mip, i+1, GLP_BV);
		}
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
	glp_load_matrix(mip, i, row_idx, col_idx, value);
	if (glp_simplex(mip, NULL) != 0) {		// solve the LP
		throw std::runtime_error("mixed_program: could not solve LP!");
	}

	// solve the integer program if needed
	if (needs_integer_step) {
		if (glp_intopt(mip, NULL) != 0) {
			throw std::runtime_error("mixed_program: could not refine LP\
				to MIP!");
		}
	}

	// return value at optimal point, and that point.
	Eigen::VectorXd x_opt(A.cols());

	for (i = 0; i < A.cols(); ++i) {
		x_opt(i) = glp_mip_col_val(mip, i+1);  		
	}

	double z = glp_mip_obj_val(mip);

	return std::pair<double, Eigen::VectorXd>(z, x_opt);
}

// To find the l_1 diameter of A, we set up the following problem:

// max sum over i: abs_i

// subject to:
//
//	for i = 1...n
//		abs_i - M * (1 - ind_abs.i) <= x_i - y_i	(1)
//		abs_i - M * ind_abs.i <= y_i - x_i			(2)
//		abs_i >= 0									(3)
//		ind_abs_i >= 0 binary						(4)
//
//	Ax <= b											(5)
//	Ay <= b											(6)

// or in matrix form:

//                                                                   b =
// x^T   = x_1 .. x_n | y_1 .. y_n | abs_1 .. abs_n | ia_1 ... ia_n || 
//
//         -----------+------------+----------------+---------------++
// (1) A =  -I        |      I     |        I       |      M * I    ||  M
//         -----------+------------+----------------+---------------++
// (2)       I        |     -I     |        I       |     -M * I    ||  0
//         -----------+------------+----------------+---------------++
// (3)       0        |      0     |       -I       |        0      ||  0
//         -----------+------------+----------------+---------------++
// (4)       0        |      0     |        0       |       -I      ||  0
//         -----------+------------+----------------+---------------++
// (5)       A        |      0     |        0       |        0      ||  b
//         -----------+------------+----------------+---------------++
// (6)       0        |      A     |        0       |        0      ||  b
//         -----------+------------+----------------+---------------++

// M must be larger than any possible x_i - y_i. In practice we just use
// some really large value.

// This can also be used to maximize the l_infinity diameter of A without
// requiring integer programming, by maximizing max i abs.i, but we don't
// do that here since what we'd really need in that case is leximax l_inf
// distance, and doing leximax with integer programming without numerical
// precision issues is hard.

struct diameter_coords {
	Eigen::VectorXd first_point;
	Eigen::VectorXd second_point;
};

diameter_coords get_extreme_coords(const polytope & poly_in, double M) {

	int n = poly_in.A.cols(), m = poly_in.A.rows();

	int prog_rows = 4 * n + 2 * m; // for (1)-(4) and (5) and (6) resp.
	int prog_cols = 4 * n;

	Eigen::MatrixXd diam_prog(prog_rows, prog_cols);
	Eigen::VectorXd diam_b(prog_rows);
	Eigen::VectorXd diam_c(prog_cols);

	// (1)
	diam_prog.block(0, 0, n, n) = -Eigen::MatrixXd::Identity(n, n);
	diam_prog.block(0, n, n, n) = Eigen::MatrixXd::Identity(n, n);
	diam_prog.block(0, 2*n, n, n) = Eigen::MatrixXd::Identity(n, n);
	diam_prog.block(0, 3*n, n, n) = M * Eigen::MatrixXd::Identity(n, n);
	// Column vector
	diam_b.block(0, 0, n, 1) = Eigen::MatrixXd::Constant(n, 1, M);
	// (2)
	diam_prog.block(n, 0, n, n) = Eigen::MatrixXd::Identity(n, n);
	diam_prog.block(n, n, n, n) = -Eigen::MatrixXd::Identity(n, n);
	diam_prog.block(n, 2*n, n, n) = Eigen::MatrixXd::Identity(n, n);
	diam_prog.block(n, 3*n, n, n) = -M * Eigen::MatrixXd::Identity(n, n);
	diam_b.block(n, 0, n, 1) = Eigen::MatrixXd::Constant(n, 1, 0);
	// (3)
	diam_prog.block(2*n, 0, n, n) = Eigen::MatrixXd::Zero(n, n);
	diam_prog.block(2*n, n, n, n) = Eigen::MatrixXd::Zero(n, n);
	diam_prog.block(2*n, 2*n, n, n) = -Eigen::MatrixXd::Identity(n, n);
	diam_prog.block(2*n, 3*n, n, n) = Eigen::MatrixXd::Zero(n, n);
	diam_b.block(2*n, 0, n, 1) = Eigen::MatrixXd::Constant(n, 1, 0);
	// (4)
	diam_prog.block(3*n, 0, n, 3*n) = Eigen::MatrixXd::Zero(n, 3*n);
	diam_prog.block(3*n, 3*n, n, n) = -Eigen::MatrixXd::Identity(n, n);
	diam_b.block(3*n, 0, n, 1) = Eigen::MatrixXd::Constant(n, 1, 0);
	// (5)
	diam_prog.block(4*n, 0, m, n) = poly_in.A;
	diam_prog.block(4*n, n, m, 3*n) = Eigen::MatrixXd::Zero(m, 3*n);
	diam_b.block(4*n, 0, m, 1) = poly_in.b;
	// (6)
	diam_prog.block(4*n+m, 0, m, n) = Eigen::MatrixXd::Zero(m, n);
	diam_prog.block(4*n+m, n, m, n) = poly_in.A;
	diam_prog.block(4*n+m, 2*n, m, 2*n) = Eigen::MatrixXd::Zero(m, 2*n);
	diam_b.block(4*n+m, 0, m, 1) = poly_in.b;

	// minimize -(abs_1 + ... + abs_n)
	diam_c.block(2*n, 0, n, 1) = Eigen::MatrixXd::Constant(n, 1, -1);

	std::vector<bool> binaries(prog_cols, false);
	for (int i = prog_cols-n; i < prog_cols; ++i) { binaries[i] = true;}

	std::pair<double, Eigen::VectorXd> output = mixed_program(diam_prog,
		diam_b, diam_c, binaries, false);

	// Split up result from the MIP solver into the actual coordinates.
	diameter_coords coords;
	coords.first_point = output.second.block(0, 0, n, 1);
	coords.second_point = output.second.block(n, 0, n, 1);
	
	return coords;
}

double get_l1_diameter(const polytope & poly_in, double M) {

	diameter_coords coords = get_extreme_coords(poly_in, M);

	return (coords.first_point - coords.second_point).lpNorm<1>();
}

// Get a lower bound on the l2 maximum-distance diameter.
double get_l2_diameter_lb(const polytope & poly_in, double M) {
	diameter_coords coords = get_extreme_coords(poly_in, M);

	// Both of these are lower bounds, and we choose the greater of the 
	// two.

	// Calculate the l2 distance between the extreme points optimizing
	// l1 distance.
	double one = (coords.first_point - coords.second_point).lpNorm<2>();

	// Use the equivalence of lp norms: ||x||_1 <= sqrt(n) * ||x||_2
	double two = (coords.first_point - coords.second_point).lpNorm<1>() /
		sqrt(coords.first_point.cols());

	return std::max(one, two);
}

main() {
	int dimension = 2;
	polytope simplex = get_simplex(dimension);

	double l_1_diam = get_l1_diameter(simplex, 1000);
	double should_be_diam = 2;

	if (should_be_diam == l_1_diam) {
		std::cout << "Test PASS" << std::endl;
	} else {
		std::cout << "Test FAIL (" << l_1_diam << ")" << std::endl;
	}

	double l_2_diam = get_l2_diameter_lb(simplex, 1000);
	double should_be = sqrt(2);

	if (should_be_diam == l_2_diam) {
		std::cout << "Test PASS" << std::endl;
	} else {
		std::cout << "Test FAIL (" << l_1_diam << ")" << std::endl;
	}
}