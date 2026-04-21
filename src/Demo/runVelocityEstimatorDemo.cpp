#include <iostream>
#include <Eigen/Dense>
#include "VelocityConsistencyEstimator.h"
#include "VelocityConsistencyEstimatorHelper.h"
#include "VelocityEstimatorExample.h"
#include "VelocityEstimatorExample1.h"
#include "VelocityEstimatorExample2.h"
#include "VelocityEstimatorExample3.h"
#include "VelocityExperiment.h"
#include "runAlgorithmsDemo.h"


void runVelocityEstimatorDemo(VelocityExperiment experiment) {
	std::cout << "Velocity Estimator" << std::endl;

	VelocityConsistencyEstimator estimator(2, 4, 2);
	Eigen::VectorXd mean_prior;
	Eigen::VectorXd mean_post;
	Eigen::MatrixXd cov_prior;
	Eigen::MatrixXd cov_post;

	switch (experiment)
	{
		case VelocityExperiment::Example0:
		{
			estimator = VelocityConsistencyEstimator(2, 4, 2);
			mean_prior = VelocityEstimatorExample::getXpred();
			mean_post = VelocityEstimatorExample::getX();
			cov_prior = VelocityEstimatorExample::getPpred();
			cov_post = VelocityEstimatorExample::getP();
			break;
		}
		case VelocityExperiment::TestIgor:
		{
			estimator = VelocityConsistencyEstimator(2, 4, 2);
			mean_prior = VelocityEstimatorExample1::getXpred();
			mean_post = VelocityEstimatorExample1::getX();
			cov_prior = VelocityEstimatorExample1::getPpred();
			cov_post = VelocityEstimatorExample1::getP();
			break;
		}
		case VelocityExperiment::TestTime83:
		{
			estimator = VelocityConsistencyEstimator(3, 9, 3);
			mean_prior = VelocityEstimatorExample2::getXpred();
			mean_post = VelocityEstimatorExample2::getX();
			cov_prior = VelocityEstimatorExample2::getPpred();
			cov_post = VelocityEstimatorExample2::getP();
			break;
		}
		case VelocityExperiment::TestTime294:
		{
			estimator = VelocityConsistencyEstimator(3, 9, 3);
			mean_prior = VelocityEstimatorExample3::getXpred();
			mean_post = VelocityEstimatorExample3::getX();
			cov_prior = VelocityEstimatorExample3::getPpred();
			cov_post = VelocityEstimatorExample3::getP();
			break;
		}
		default:
		{
			throw std::invalid_argument("Unknown VelocityExperiment value");
		}
	}

	int N = 1000;

	try {
		auto ltrRes = estimator.computeTrackLikelihood(N, mean_post, cov_post, mean_prior, cov_prior);

		std::cout << "L_tr_kde = " << ltrRes.kde << std::endl;
		std::cout << "L_tr_gaussian = " << ltrRes.gaussian << std::endl;

		std::cout << "L_tr_kde_weighted = " << ltrRes.kde_weighted << std::endl;
		std::cout << "L_tr_gaussian_weighted = " << ltrRes.gaussian_weighted << std::endl;
		//std::cout << "EstimateIntegral = " << ltrRes.integral << std::endl;
		//std::cout << "EstimateIntegralWeighted = " << ltrRes.integral_weighted << std::endl;
		std::cout << "L_tr_kdeResampled = " << ltrRes.kde_resampled << std::endl;

		std::cout << "L_tr_kernelSimilarity = " << ltrRes.kernelSimilarity << std::endl;
		std::cout << "L_tr_kernelSimilarity_weighted = " << ltrRes.kernelSimilarity_weighted << std::endl;


		const Eigen::VectorXd& v_post = estimator.getPosteriorSpeeds();
		const Eigen::VectorXd& v_prior = estimator.getPriorSpeeds();
		const Eigen::MatrixXd& v_selector = estimator.getVelocitySelector();

		std::cout << "mean v_prior: " << v_prior.mean() << std::endl;
		std::cout << "mean v_post: " << v_post.mean() << std::endl;

		double v_min = std::min(v_prior.minCoeff(), v_post.minCoeff());
		double v_max = std::max(v_prior.maxCoeff(), v_post.maxCoeff());

		//VelocityConsistencyEstimatorHelper::exportKdeCurve(v_prior, v_min, v_max, 100, "kde_prior.csv");
		//VelocityConsistencyEstimatorHelper::exportKdeCurve(v_post, v_min, v_max, 100, "kde_post.csv");

		std::vector<double> x_prior, kde_prior;
		VelocityConsistencyEstimatorHelper::computeKdeCurve(v_prior, v_min, v_max, 100, x_prior, kde_prior, "kde_prior.csv");

		std::vector<double> x_post, kde_post;
		VelocityConsistencyEstimatorHelper::computeKdeCurve(v_post, v_min, v_max, 100, x_post, kde_post, "kde_post.csv");

		std::vector<double> gaussian_x, gaussian_prior;
		VelocityConsistencyEstimatorHelper::getGaussianCurve(v_selector, mean_prior, cov_prior, v_min, v_max, 100, gaussian_x, gaussian_prior, "gaussian_prior.csv");

		runPlotPython(x_post, kde_post, kde_prior, gaussian_prior, v_min, v_max, ltrRes.kde);

		//VelocityConsistencyEstimatorHelper::exportGaussianCurve(v_selector, mean_prior, cov_prior, v_min, v_max, 100, "gaussian_prior.csv");
	}

	catch (const std::exception& e) {
		std::cerr << "Error while estimating L_tr: " << e.what() << std::endl;
	}
}
