#pragma once
#include <eigen3/Eigen/Dense>

// defined by Ax <= b
struct polytope {
	Eigen::MatrixXd A;
	Eigen::VectorXd b;
};
