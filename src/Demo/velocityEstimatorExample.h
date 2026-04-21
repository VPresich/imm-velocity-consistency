#pragma once
#include <Eigen/Dense>

namespace VelocityEstimatorExample {

	inline Eigen::VectorXd getXpred() {
		Eigen::VectorXd vec(4);
		vec << 0, 0, 1.0, 1.0;
		return vec;
	}


	inline Eigen::VectorXd getX() {
		Eigen::VectorXd vec(4);
		vec << 0, 0, 1.0, 1.0;
		return vec;
	}


	inline Eigen::MatrixXd getPpred() {
		Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(4, 4);
		mat.diagonal() << 0.1, 0.1, 0.1, 0.1;
		return mat;
	}


	inline Eigen::MatrixXd getP() {
		Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(4, 4);
		mat.diagonal() << 0.05, 0.05, 0.1, 0.1;
		return mat;
	}

}
