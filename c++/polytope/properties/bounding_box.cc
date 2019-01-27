#include "bounding_box.h"
#include "../simplex.h"

#include <iostream>

std::pair<Eigen::VectorXd, Eigen::VectorXd> polytope_bounding_box::
	get_bounding_box(const polytope & poly_in) const {

	int dimension = poly_in.get_dimension();

	// Solve the LPs min x_1, x_2, x_3... to get the left end of the
	// box.

	Eigen::VectorXd x_min(dimension), x_max(dimension);
	Eigen::VectorXd objective = Eigen::VectorXd::Zero(dimension);
	int i;

	for (i = 0; i < dimension; ++i) {
		objective(i) = 1;
		x_min(i) = poly_in.linear_program(objective, false).second(i);
		objective(i) = 0;
	}

	// Solve the LPs max x_1, x_2, x_3... to get the right end of the
	// box.

	for (i = 0; i < dimension; ++i) {
		objective(i) = -1;
		x_max(i) = poly_in.linear_program(objective, false).second(i);
		objective(i) = 0;
	}

	return std::pair<Eigen::VectorXd, Eigen::VectorXd>(x_min, x_max);
}

double polytope_bounding_box::get_max_axis_length(
	const polytope & poly_in) const {

	std::pair<Eigen::VectorXd, Eigen::VectorXd> box = get_bounding_box(
		poly_in);

	// Every value will be nonnegative, so the l_inf norm is the maximum
	// element, which gives the length of the bounding box along the
	// widest axis.
	return (box.second-box.first).lpNorm<Eigen::Infinity>();
}

#ifdef TEST_BB

main() {
	simplex u_simplex(2);

	std::pair<Eigen::VectorXd, Eigen::VectorXd> box =
		polytope_bounding_box().get_bounding_box(u_simplex);

	std::pair<Eigen::Vector2d, Eigen::Vector2d> should_be;
	should_be.first << 0, 0;
	should_be.second << 1, 1;

	if (box.first == should_be.first &&
		box.second == should_be.second) {
		std::cout << "Test PASS" << std::endl;
	} else {
		std::cout << "Test FAIL" << std::endl;
	}

	if (polytope_bounding_box().get_max_axis_length(u_simplex) == 1) {
		std::cout << "Test PASS" << std::endl;
	} else {
		std::cout << "Test FAIL" << std::endl;
	}
}

#endif