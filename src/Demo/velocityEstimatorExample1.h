#pragma once
#include <Eigen/Dense>

// Igor Example

namespace VelocityEstimatorExample1 {

	inline Eigen::VectorXd getXpred() {
		Eigen::VectorXd vec(4);
		vec << 0, 0, 50, 87;
		return vec;
	}

	inline Eigen::VectorXd getX() {
		Eigen::VectorXd vec(4);
		vec << 0, 0, 50, 87;
		return vec;
	}


	inline Eigen::MatrixXd getPpred() {
		Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(4, 4);
		mat.diagonal() << 0.05, 0.05, 25., 4.;
		mat(2, 3) = -5.;
		mat(3, 2) = -5.;
		return mat;
	}


	inline Eigen::MatrixXd getP() {
		Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(4, 4);
		mat.diagonal() << 0.05, 0.05, 25., 4.;
		mat(2, 3) = -5.;
		mat(3, 2) = -5.;
		return mat;
	}
}
