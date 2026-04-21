#pragma once
#include <vector>
#include <Eigen/Dense>

class VelocityConsistencyEstimatorHelper
{
public:
	static double bandwidthSilverman(const Eigen::VectorXd& data);
	static double computeCombinedStd(const Eigen::VectorXd& v_post, const Eigen::VectorXd& v_prior);
	static double kdeAtPoint(double x, const Eigen::VectorXd& data, double h, int excludeIndex = -1);
	static double kernel(double a, double b, double sigma2);
	static double weightedKdeAtPoint(double x, const Eigen::VectorXd& data, const Eigen::VectorXd& weights, double h);
	static double normalPdf(double x, double mean, double sigma);
	static std::pair<double, double> estimateSpeedStats(const Eigen::MatrixXd& selector, const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov);
	static double multivariateNormalPdf(const Eigen::VectorXd& x,	const Eigen::VectorXd& mean, const Eigen::LLT<Eigen::MatrixXd>& llt);

	static void exportKdeCurve(const Eigen::VectorXd& v_data, double speed_min, double speed_max, int num_points, const std::string& filename);
	static void computeKdeCurve(const Eigen::VectorXd& data, double min, double max, int num_points, std::vector<double>& x_out, std::vector<double>& y_out, const std::string& filename);
	static void exportGaussianCurve(const Eigen::MatrixXd& v_selector, const Eigen::VectorXd& x, const Eigen::MatrixXd& P, double speed_min, double speed_max, int num_points, const std::string& filename);
	static void getGaussianCurve(const Eigen::MatrixXd& v_selector, const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov,
		double speed_min, double speed_max,	int num_points,	std::vector<double>& out_x,	std::vector<double>& out_y,	const std::string& filename );
};
