#pragma once
#include <eigen3/Eigen/Dense>
#include <vector>

// Convex polytope defined by Ax <= b. This is a base class that
// permits reading but not writing; and even reading is pure virtual.

class polytope {
	public:
		// Virtual so that complex polytope can update A and b from
		// other matrices if they've gone stale.
		virtual const Eigen::MatrixXd & get_A() const = 0;
		virtual const Eigen::VectorXd & get_b() const = 0;

		virtual size_t get_num_halfplanes() const { 
			return get_A().rows(); }
		virtual size_t get_dimension() const { return get_A().cols(); }

		// Solves a mixed integer program
		// min cTx s.t. Ax <= b
		// with x_i as binary for all i for which is_binary[i] is true.

		virtual std::pair<double, Eigen::VectorXd> mixed_program(
			const Eigen::VectorXd & c, const std::vector<bool> & is_binary,
			bool verbose) const;

		// Solves the linear program
		// min cTx s.t. Ax <= b

		virtual std::pair<double, Eigen::VectorXd> linear_program(
			const Eigen::VectorXd & c, 	bool verbose) const;

		bool is_inside(const Eigen::VectorXd & x) const;

};