#include <iostream>
#include <eigen3/Eigen/Dense>
#include <glpk.h>

// Represents the polytope AFz <= bn, with x = Fz + x_0.

struct equality_reduction {
	Eigen::MatrixXd AF;
	Eigen::VectorXd bn;
	Eigen::MatrixXd F;
	Eigen::VectorXd x_0;
};

// Input: The convex polytope Ax <= b, Cx = d.
// Output: A polytope of smaller dimension, defined by inequalities alone,
//			where any point inside the polytope can be turned into
//			a point in the input polytope by linear transformation.

equality_reduction get_equality_reduction(const Eigen::MatrixXd & A,
	const Eigen::VectorXd & b, const Eigen::MatrixXd & C,
	const Eigen::VectorXd & d) {

	// To perform the reduction from inequality-and-equality polytope
	// to inequality-only polytope, we construct F and x_0 so that for
	// any vector z, x = Fz + x_0 satisfies Cx = d. Then we simply insert
	// this definition into the output polytope to get

	// Ax <= b, Cx = d		--> 	A(Fz + x_0) <= b
	//						-->		AFz + Ax_0 <= b
	//						-->		AFz <= b - Ax_0

	// thus letting AF = A*F, bn = b - Ax_0 works.

	// To find F and x_0, we observe that if we let x_0 be a solution
	// (any solution) to Cx = d, and if Fz is always a solution to
	// Cx = 0, then x = Fz + x_0 will also always be a solution to
	// Cx = d. Setting F to the kernel of the matrix C thus works
	// nicely.

	Eigen::FullPivLU<Eigen::MatrixXd> lu(C);

	equality_reduction output;

	output.F = lu.kernel();			// The null space of C.
	output.x_0 = lu.solve(d);		// It doesn't matter what solution.
	output.bn = b - A * output.x_0;
	output.AF = A * output.F;

	return output;
}

// minimize cTx subject to Ax <= b. All variables are unconstrained.
// GLPK syntax is ugly.

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

/*
 Example:

 A = |-1  0  0  0 | b = |0|  C = |1 1 1 0| d = | 50|
     | 0 -1  0  0 |     |0|      |2 3 4 0|     |158|
     | 0  0 -1  0 |     |0|      |4 3 2 0|     |142|
     | 0  0  0 -1 |     |0|
     | 0  0  0  1 |     |1|
*/


int main()
{
	typedef Eigen::Matrix<double, 5, 4> Matrix5x4;
	typedef Eigen::Matrix<double, 3, 4> Matrix3x4;
	typedef Eigen::Matrix<double, 3, 3> Matrix3x3;

	Matrix3x4 C;
	Matrix5x4 A;
	Eigen::Vector3d d;
	Eigen::VectorXd b(5);

	A << -1, 0, 0, 0, \
		 0, -1, 0, 0, \
		 0, 0, -1, 0, \
		 0, 0, 0, -1, \
		 0, 0, 0, 1;

	b << 0, 0, 0, 0, 1;

	C << 1, 1, 1, 0, \
		 2, 3, 4, 0, \
		 4, 3, 2, 0;

	d << 50, 158, 142;

	equality_reduction reduction = get_equality_reduction(A, b, C, d);

	std::cout << "A =\n" << A << "\n" << std::endl;
	std::cout << "b =\n" << b << "\n" << std::endl;
	std::cout << "C =\n" << C << "\n" << std::endl;
	std::cout << "d =\n" << d << "\n" << std::endl;

	std::cout << "---" << std::endl;

	std::cout << "AF=\n" << reduction.AF << "\n" << std::endl;
	std::cout << "bn=\n" << reduction.bn << "\n" << std::endl;
	std::cout << "F =\n" << reduction.F << "\n" << std::endl;
	std::cout << "x_0 =\n" << reduction.x_0 << "\n" << std::endl;

	std::cout << "---" << std::endl;

	// LP test, kinda kludgy
	// min cTx s.t. Ax <= b, Cx = d.
	// We know the classical solution is (0, 42, 8, 0), so let's try it
	// with a reduced A (i.e. equality constraints folded in).

	Eigen::Vector4d c;
	c << 0, -1, 0, 0;

  	std::pair<double, Eigen::VectorXd> opt = linear_program(reduction.AF,
  		reduction.bn, c.transpose() * reduction.F);

  	Eigen::VectorXd x_opt = reduction.F * opt.second + reduction.x_0;
  	Eigen::Vector4d should_be;
  	should_be << 0, 42, 8, 0;

  	std::cout << x_opt << std::endl;

  	if (x_opt == should_be) {
  		std::cout << "Test PASS" << std::endl;
  	} else {
  		std::cout << "Test FAIL" << std::endl;
  	}
}