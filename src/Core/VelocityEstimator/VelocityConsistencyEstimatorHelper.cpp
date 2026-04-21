#include "VelocityConsistencyEstimatorHelper.h"
#include "EstimatorConstants.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <iostream>
#include <fstream>


double VelocityConsistencyEstimatorHelper::bandwidthSilverman(const Eigen::VectorXd& data) {
	int n = static_cast<int>(data.size());
	if (n < 2) {
		throw std::invalid_argument("Need at least two data points to compute Silverman's bandwidth.");
	}

	double mean = data.mean();
	double stddev = std::sqrt((data.array() - mean).square().sum() / (n - 1));

	if (stddev == 0.0) {
		return std::numeric_limits<double>::epsilon();
	}

	return 1.06 * stddev * std::pow(n, -0.2);
}

double VelocityConsistencyEstimatorHelper::computeCombinedStd(const Eigen::VectorXd& v_post, const Eigen::VectorXd& v_prior)
{
	int n_post = static_cast<int>(v_post.size());
	int n_prior = static_cast<int>(v_prior.size());
	int n = n_post + n_prior;

	if (n < 2) {
		throw std::invalid_argument("Need at least two data points to compute std.");
	}

	Eigen::VectorXd all(n);
	all << v_post, v_prior;

	double mean = all.mean();

	double stddev = std::sqrt((all.array() - mean).square().sum() / (n - 1));

	if (stddev == 0.0) {
		return std::numeric_limits<double>::epsilon();
	}

	return stddev;
}


double VelocityConsistencyEstimatorHelper::kdeAtPoint(double x, const Eigen::VectorXd& data, double h, int excludeIndex) {
	int N = static_cast<int>(data.size());

	// 1. Считаем сумму экспонент для всех элементов через Eigen (векторизованно)
	double sum = ((data.array() - x).square() / (-2.0 * h * h)).exp().sum();

	// 2. Если задан корректный индекс, вычитаем вклад этой точки
	int effectiveN = N;
	if (excludeIndex >= 0 && excludeIndex < N) {
		double diff = data[excludeIndex] - x;
		double selfContribution = std::exp((diff * diff) / (-2.0 * h * h));
		sum -= selfContribution;
		effectiveN = N - 1; // Уменьшаем количество точек для усреднения
	}

	// 3. Считаем итоговый коэффициент (нормировка Гауссианы и усреднение)
	if (effectiveN <= 0) return 0.0;

	double coeff = 1.0 / (effectiveN * h * std::sqrt(2.0 * M_PI));
	return coeff * sum;
}


double VelocityConsistencyEstimatorHelper::weightedKdeAtPoint(double x, const Eigen::VectorXd& data, const Eigen::VectorXd& weights, double h) {
	double coeff = 1.0 / (h * std::sqrt(2.0 * M_PI));
	double sum = 0.0;
	for (int i = 0; i < data.size(); ++i) {
		double u = (x - data[i]) / h;
		sum += weights[i] * std::exp(-0.5 * u * u);
	}
	return coeff * sum;
}


double VelocityConsistencyEstimatorHelper::normalPdf(double x, double mean, double sigma) {
	const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * M_PI);
	double z = (x - mean) / sigma;
	return (inv_sqrt_2pi / sigma) * std::exp(-0.5 * z * z);
}


double VelocityConsistencyEstimatorHelper::multivariateNormalPdf(
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& mean,
	const Eigen::LLT<Eigen::MatrixXd>& llt)
{
	const int dim = static_cast<int>(x.size());
	const Eigen::MatrixXd& L = llt.matrixL();

	Eigen::VectorXd diff = x - mean;
	Eigen::VectorXd solved = llt.solve(diff);
	double exponent = -0.5 * diff.dot(solved);
	double det_sqrt = L.diagonal().prod();
	double norm_const = 1.0 / (std::pow(2 * M_PI, dim/2) * det_sqrt);

	return norm_const * std::exp(exponent);
}


std::pair<double, double> VelocityConsistencyEstimatorHelper::estimateSpeedStats(
	const Eigen::MatrixXd& selector,
	const Eigen::VectorXd& mean,
	const Eigen::MatrixXd& cov
) {
	Eigen::VectorXd velocity = selector * mean;
	double velocityNorm = velocity.norm();
	Eigen::MatrixXd velocityCov = selector * cov * selector.transpose();

	Eigen::VectorXd velocityGrad;
	if (velocityNorm < EstimatorConstants::kVelocityNorm) {
		velocityGrad.setZero();
	}
	else {
		velocityGrad = velocity / velocityNorm;
	}

	double velocityVar = velocityGrad.transpose() * velocityCov * velocityGrad;

	return std::make_pair(velocityNorm, std::sqrt(velocityVar));
}


void VelocityConsistencyEstimatorHelper::exportKdeCurve(const Eigen::VectorXd& data, double min,
	double max, int num_points, const std::string& filename) {

	double h = VelocityConsistencyEstimatorHelper::bandwidthSilverman(data);

	std::ofstream out(filename);
	if (!out) {
		std::cerr << "Cannot open file: " << filename << std::endl;
		return;
	}

	double step = (max - min) / (num_points - 1);
	for (int i = 0; i < num_points; ++i) {
		double speed = min + i * step;
		double kde_val = VelocityConsistencyEstimatorHelper::kdeAtPoint(speed, data, h);
		out << speed << ";" << kde_val << "\n";
	}

	out.close();
	std::cout << "KDE written to: " << filename << std::endl;
}

void VelocityConsistencyEstimatorHelper::computeKdeCurve(
	const Eigen::VectorXd& data,
	double min,
	double max,
	int num_points,
	std::vector<double>& x_out,
	std::vector<double>& y_out,
	const std::string& filename = "")
{
	if (data.size() == 0 || num_points < 2)
		throw std::invalid_argument("Invalid data or num_points");

	double h = VelocityConsistencyEstimatorHelper::bandwidthSilverman(data);

	x_out.resize(num_points);
	y_out.resize(num_points);

	double step = (max - min) / (num_points - 1);

	std::ofstream out;
	if (!filename.empty()) {
		out.open(filename);
		if (!out)
			throw std::runtime_error("Cannot open file: " + filename);
	}

	for (int i = 0; i < num_points; ++i)
	{
		double speed = min + i * step;
		double kde_val =
			VelocityConsistencyEstimatorHelper::kdeAtPoint(speed, data, h);

		x_out[i] = speed;
		y_out[i] = kde_val;

		if (out.is_open())
			out << speed << ";" << kde_val << "\n";
	}

	if (out.is_open()) {
		out.close();
		std::cout << "KDE written to: " << filename << std::endl;
	}
}


void VelocityConsistencyEstimatorHelper::exportGaussianCurve(
	const Eigen::MatrixXd& v_selector,
	const Eigen::VectorXd& x,
	const Eigen::MatrixXd& P,
	double speed_min,
	double speed_max,
	int num_points,
	const std::string& filename
) {
	std::pair<double, double> speed_stats = VelocityConsistencyEstimatorHelper::estimateSpeedStats(v_selector, x, P);
	double mu_speed = speed_stats.first;
	double sigma_speed = speed_stats.second;

	std::ofstream out(filename);
	if (!out) {
		std::cerr << "Cannot open file: " << filename << std::endl;
		return;
	}

	double step = (speed_max - speed_min) / (num_points - 1);
	for (int i = 0; i < num_points; ++i) {
		double speed = speed_min + i * step;
		double gaussian_val = VelocityConsistencyEstimatorHelper::normalPdf(speed, mu_speed, sigma_speed);
		out << speed << ";" << gaussian_val << "\n";
	}

	out.close();
	std::cout << "Gaussian curve written to: " << filename << std::endl;
}

void VelocityConsistencyEstimatorHelper::getGaussianCurve(
	const Eigen::MatrixXd& v_selector, // Добавил селектор в аргументы
	const Eigen::VectorXd& mean,
	const Eigen::MatrixXd& cov,
	double speed_min,
	double speed_max,
	int num_points,
	std::vector<double>& out_x,
	std::vector<double>& out_y,
	const std::string& filename // Путь к файлу
) {

	auto stats = VelocityConsistencyEstimatorHelper::estimateSpeedStats(v_selector, mean, cov);
	double mu = stats.first;
	double sigma = stats.second;

	// 2. Подготовка файла для записи
	std::ofstream out(filename);
	if (!out) {
		std::cerr << "Warning: Cannot open file for Gaussian curve: " << filename << std::endl;
	}

	// 3. Генерация кривой
	out_x.clear();
	out_y.clear();
	out_x.reserve(num_points);
	out_y.reserve(num_points);

	double step = (speed_max - speed_min) / (num_points - 1);

	for (int i = 0; i < num_points; ++i) {
		double v = speed_min + i * step;
		double pdf_val = normalPdf(v, mu, sigma);

		// В память (для Python)
		out_x.push_back(v);
		out_y.push_back(pdf_val);

		// В файл (для CSV)
		if (out.is_open()) {
			out << v << ";" << pdf_val << "\n";
		}
	}

	if (out.is_open()) {
		out.close();
		std::cout << "Gaussian curve also written to: " << filename << std::endl;
	}
}

double VelocityConsistencyEstimatorHelper::kernel(double a, double b, double sigma2) {
	double diff = a - b;
	double norm_const = 1.0 / std::sqrt(2.0 * M_PI * sigma2);
	return norm_const * std::exp(-(diff * diff) / (2.0 * sigma2));
}
