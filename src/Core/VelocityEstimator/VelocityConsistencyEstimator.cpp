#include <iostream>
#include <fstream>
#include <iomanip>
#include "VelocityConsistencyEstimator.h"
#include "VelocityConsistencyEstimatorHelper.h"
#include "EstimatorConstants.h"
#include <cmath>

VelocityConsistencyEstimator::VelocityConsistencyEstimator(int vel_dim, int st_dim, int vel_offset)
	: randomGenerator(std::random_device{}())
	, normalDistribution(0.0, 1.0)
	, velocityDim(vel_dim)
	, stateDim(st_dim)
	, velocityOffset(vel_offset) {
	initializeVelocitySelector();
}


void VelocityConsistencyEstimator::initializeVelocitySelector() {
	velocitySelector = Eigen::MatrixXd::Zero(velocityDim, stateDim);
	for (int i = 0; i < velocityDim; ++i) {
		velocitySelector(i, velocityOffset + i) = 1.0;
	}
}


Eigen::MatrixXd VelocityConsistencyEstimator::sampleFromGaussianLlt(
	int N, const Eigen::VectorXd& mean, const Eigen::LLT<Eigen::MatrixXd>& llt)
{
	const int dim = static_cast<int>(mean.size());
	Eigen::MatrixXd L = llt.matrixL();

	Eigen::MatrixXd Z = Eigen::MatrixXd::NullaryExpr(dim, N, [&]() { return normalDistribution(randomGenerator); });
	Eigen::MatrixXd samples = (L * Z).transpose();
	samples.rowwise() += mean.transpose();
	return samples;
}


Eigen::VectorXd VelocityConsistencyEstimator::computeGaussianPdfWeights(
	const Eigen::MatrixXd& samples,
	const Eigen::VectorXd& mean,
	const Eigen::LLT<Eigen::MatrixXd>& llt
) {
	const int N = static_cast<int>(samples.rows());
	Eigen::VectorXd weights(N);

	for (int i = 0; i < N; ++i) {
		Eigen::VectorXd x = samples.row(i).transpose();
		weights[i] = VelocityConsistencyEstimatorHelper::multivariateNormalPdf(x, mean, llt);
	}

	double total_weight = weights.sum();
	if (total_weight > 0.0)
		weights /= total_weight;

	return weights;
}


Eigen::VectorXd VelocityConsistencyEstimator::computeVelocityNorms(const Eigen::MatrixXd& samples) {
	Eigen::MatrixXd V = samples * velocitySelector.transpose();

	return V.rowwise().norm();
}


double VelocityConsistencyEstimator::computeKde() {
	if (bandwidthPriorSpeeds < 0) {
		throw std::runtime_error("Bandwidth not initialized");
	}
	double L_tr = 0.0;
	for (int i = 0; i < posteriorSpeeds.size(); ++i) {
		L_tr += VelocityConsistencyEstimatorHelper::kdeAtPoint(posteriorSpeeds[i], priorSpeeds, bandwidthPriorSpeeds);
	}
	return L_tr / posteriorSpeeds.size();
}


double VelocityConsistencyEstimator::computeGaussian()
{
	if (sigmaPriorSpeed <= 0) {
		throw std::runtime_error("sigmaPriorSpeed must be positive.");
	}

	if (meanPriorSpeed < 0) {
		throw std::runtime_error("meanPriorSpeed is negative or uninitialized");
	}

	double L_tr = 0.0;
	for (int i = 0; i < posteriorSpeeds.size(); ++i) {
		L_tr += VelocityConsistencyEstimatorHelper::normalPdf(posteriorSpeeds[i], meanPriorSpeed, sigmaPriorSpeed);
	}

	return L_tr / posteriorSpeeds.size();
}


double VelocityConsistencyEstimator::computeKdeWeighted() {

	if (bandwidthPriorSpeeds < 0) {
		throw std::runtime_error("Bandwidth not initialized");
	}

	if (posteriorSpeeds.size() != posteriorWeights.size()) {
		throw std::runtime_error("posteriorSpeeds and posteriorWeights must have the same size.");
	}
	double L_tr = 0.0;
	for (int i = 0; i < posteriorSpeeds.size(); ++i) {
		L_tr += posteriorWeights[i] * VelocityConsistencyEstimatorHelper::kdeAtPoint(posteriorSpeeds[i], priorSpeeds, bandwidthPriorSpeeds);
	}
	return L_tr;
}

double VelocityConsistencyEstimator::computeGaussianWeighted()
{
	if (posteriorSpeeds.size() != posteriorWeights.size()) {
		throw std::runtime_error("posteriorSpeeds and posteriorWeights must have the same size.");
	}

	if (sigmaPriorSpeed <= 0) {
		throw std::runtime_error("sigmaPriorSpeed must be positive.");
	}

	if (meanPriorSpeed < 0) {
		throw std::runtime_error("meanPriorSpeed is negative or uninitialized");
	}

	double L_tr = 0.0;
	for (int i = 0; i < posteriorSpeeds.size(); ++i) {
		L_tr += posteriorWeights[i] * VelocityConsistencyEstimatorHelper::normalPdf(posteriorSpeeds[i], meanPriorSpeed, sigmaPriorSpeed);
	}

	return L_tr;
}

double VelocityConsistencyEstimator::computeKernelSimilarityWeighted()
{
	if (combinedStd <= 0.0) {
		throw std::runtime_error("CombinedStd not initialized");
	}

	if (posteriorSpeeds.size() == 0 || priorSpeeds.size() == 0) {
		throw std::runtime_error("Empty speed arrays");
	}

	if (posteriorWeights.size() == 0 || priorWeights.size() == 0) {
		throw std::runtime_error("Empty weights arrays");
	}

	if (posteriorSpeeds.size() != posteriorWeights.size()) {
		throw std::runtime_error("posteriorSpeeds and posteriorWeights must have the same size.");
	}

	if (priorSpeeds.size() != priorWeights.size()) {
		throw std::runtime_error("priorSpeeds and priorWeights must have the same size.");
	}

	double sigma2 = combinedStd * combinedStd;

	double L_tr = 0.0;
	for (int i = 0; i < posteriorSpeeds.size(); ++i) {
		for (int j = 0; j < priorSpeeds.size(); ++j) {
			L_tr += posteriorWeights[i] * priorWeights[j] * VelocityConsistencyEstimatorHelper::kernel(posteriorSpeeds[i], priorSpeeds[j], sigma2);
		}
	}

	return L_tr;
}


double VelocityConsistencyEstimator::computeNormalizationDenominator() {

	if (sigmaPostSpeed <= 0) {
		throw std::runtime_error("sigmaPostSpeed must be positive.");
	}

	if (meanPostSpeed < 0) {
		throw std::runtime_error("meanPostSpeed is negative or uninitialized");
	}

	if (sigmaPriorSpeed <= 0) {
		throw std::runtime_error("sigmaPriorSpeed must be positive.");
	}

	if (meanPriorSpeed < 0) {
		throw std::runtime_error("meanPriorSpeed is negative or uninitialized");
	}

	double sum_post = 0.0;
	for (int i = 0; i < posteriorSpeeds.size(); ++i) {
		sum_post += VelocityConsistencyEstimatorHelper::normalPdf(
			posteriorSpeeds[i], meanPostSpeed, sigmaPostSpeed);
	}
	sum_post /= posteriorSpeeds.size();

	double sum_prior = 0.0;
	for (int i = 0; i < priorSpeeds.size(); ++i) {
		sum_prior += VelocityConsistencyEstimatorHelper::normalPdf(
			priorSpeeds[i], meanPriorSpeed, sigmaPriorSpeed);
	}
	sum_prior /= priorSpeeds.size();

	return std::sqrt(sum_post * sum_prior);
}


double VelocityConsistencyEstimator::computeNormalizationDenominatorKde() {

	if (bandwidthPriorSpeeds < 0) {
		throw std::runtime_error("bandwidthPriorSpeeds not initialized");
	}

	if (bandwidthPostSpeeds < 0) {
		throw std::runtime_error("bandwidthPostSpeeds not initialized");
	}

	double sum_post = 0.0;
	for (int i = 0; i < posteriorSpeeds.size(); ++i) {
		sum_post += VelocityConsistencyEstimatorHelper::kdeAtPoint(
			posteriorSpeeds[i], posteriorSpeeds, bandwidthPostSpeeds, i);
	}
	sum_post /= posteriorSpeeds.size();

	double sum_prior = 0.0;
	for (int i = 0; i < priorSpeeds.size(); ++i) {
		sum_prior += VelocityConsistencyEstimatorHelper::kdeAtPoint(
			priorSpeeds[i], priorSpeeds, bandwidthPriorSpeeds, i);
	}
	sum_prior /= priorSpeeds.size();

	return std::sqrt(sum_post * sum_prior);
}

double VelocityConsistencyEstimator::computeKernelSimilarityDenominatorWeighted()
{
	if (combinedStd <= 0.0) {
		throw std::runtime_error("combinedStd not initialized");
	}

	if (posteriorSpeeds.size() == 0 || priorSpeeds.size() == 0) {
		throw std::runtime_error("Empty speed arrays");
	}

	if (posteriorWeights.size() == 0 || priorWeights.size() == 0) {
		throw std::runtime_error("Empty weights arrays");
	}

	if (posteriorSpeeds.size() != posteriorWeights.size()) {
		throw std::runtime_error("posteriorSpeeds and posteriorWeights must have the same size.");
	}

	if (priorSpeeds.size() != priorWeights.size()) {
		throw std::runtime_error("priorSpeeds and priorWeights must have the same size.");
	}

	double sigma2 = combinedStd * combinedStd;

	double self_post = 0.0;
	int N_post = static_cast<int>(posteriorSpeeds.size());

	for (int i = 0; i < N_post; ++i) {
		for (int k = 0; k < N_post; ++k) {
			double w_prod = posteriorWeights[i] * posteriorWeights[k];
			self_post += w_prod * VelocityConsistencyEstimatorHelper::kernel(posteriorSpeeds[i], posteriorSpeeds[k], sigma2);
		}
	}

	double self_prior = 0.0;
	int N_prior = static_cast<int>(priorSpeeds.size());

	for (int j = 0; j < N_prior; ++j) {
		for (int l = 0; l < N_prior; ++l) {
			double w_prod = priorWeights[j] * priorWeights[l];
			self_prior += w_prod * VelocityConsistencyEstimatorHelper::kernel(priorSpeeds[j], priorSpeeds[l], sigma2);
		}
	}

	return std::sqrt(self_post * self_prior);
}


double VelocityConsistencyEstimator::computeKernelSimilarity()
{
	if (combinedStd <= 0.0) {
		throw std::runtime_error("combinedStd not initialized");
	}

	if (posteriorSpeeds.size() == 0 || priorSpeeds.size() == 0) {
		throw std::runtime_error("Empty speed arrays");
	}

	double sigma2 = combinedStd * combinedStd;

	// числитель: среднее по всем парам post/prior
	double L_tr = 0.0;
	for (int i = 0; i < posteriorSpeeds.size(); ++i) {
		for (int j = 0; j < priorSpeeds.size(); ++j) {
			L_tr += VelocityConsistencyEstimatorHelper::kernel(posteriorSpeeds[i], priorSpeeds[j], sigma2);
		}
	}
	L_tr /= static_cast<double>(posteriorSpeeds.size() * priorSpeeds.size());

	return L_tr;
}

double VelocityConsistencyEstimator::computeKernelSimilarityDenominator()
{
	if (combinedStd <= 0.0) {
		throw std::runtime_error("combinedStd not initialized");
	}

	if (posteriorSpeeds.size() == 0 || priorSpeeds.size() == 0) {
		throw std::runtime_error("Empty speed arrays");
	}

	int N_prior = static_cast<int>(priorSpeeds.size());
	int N_post = static_cast<int>(posteriorSpeeds.size());

	if (N_post < 2 || N_prior < 2) {
		throw std::runtime_error("Need at least 2 points for unbiased estimate");
	}

	double sigma2 = combinedStd * combinedStd;

	// self-overlap posterior
	double self_post = 0.0;
	for (int i = 0; i < N_post; ++i) {
		for (int k = 0; k < N_post; ++k) {
			if (i == k) continue;
			self_post += VelocityConsistencyEstimatorHelper::kernel(posteriorSpeeds[i], posteriorSpeeds[k], sigma2);
		}
	}
	//self_post /= static_cast<double>(N_post * N_post);
	self_post /= static_cast<double>(N_post * (N_post - 1));

	// self-overlap prior
	double self_prior = 0.0;
	for (int j = 0; j < N_prior; ++j) {
		for (int l = 0; l < N_prior; ++l) {
			if (j == l) continue;
			self_prior += VelocityConsistencyEstimatorHelper::kernel(priorSpeeds[j], priorSpeeds[l], sigma2);
		}
	}
	//self_prior /= static_cast<double>(N_prior * N_prior);
	self_prior /= static_cast<double>(N_prior * (N_prior - 1));

	return std::sqrt(self_post * self_prior);
}

double VelocityConsistencyEstimator::computeKdeInnerProductWeighted()
{
	if (bandwidthPriorSpeeds <= 0.0 || bandwidthPostSpeeds <= 0.0) {
		throw std::runtime_error("Bandwidths not initialized");
	}

	if (posteriorSpeeds.size() == 0 || priorSpeeds.size() == 0) {
		throw std::runtime_error("Empty speed arrays");
	}

	const int N_post = static_cast<int>(posteriorSpeeds.size());
	const int N_prior = static_cast<int>(priorSpeeds.size());

	double sigma2 = bandwidthPriorSpeeds * bandwidthPriorSpeeds +
		bandwidthPostSpeeds * bandwidthPostSpeeds;

	double sum = 0.0;

	for (int i = 0; i < N_prior; ++i) {
		for (int j = 0; j < N_post; ++j) {
			sum += VelocityConsistencyEstimatorHelper::kernel(
				priorSpeeds[i],
				posteriorSpeeds[j],
				sigma2
			);
		}
	}

	return sum / static_cast<double>(N_prior * N_post);
}


double VelocityConsistencyEstimator::computeKdeInnerProduct()
{
	if (bandwidthPriorSpeeds <= 0.0 || bandwidthPostSpeeds <= 0.0) {
		throw std::runtime_error("Bandwidths not initialized");
	}

	if (posteriorSpeeds.size() == 0 || priorSpeeds.size() == 0) {
		throw std::runtime_error("Empty speed arrays");
	}

	const int N_post = static_cast<int>(posteriorSpeeds.size());
	const int N_prior = static_cast<int>(priorSpeeds.size());

	double sigma2 = bandwidthPriorSpeeds * bandwidthPriorSpeeds +
		bandwidthPostSpeeds * bandwidthPostSpeeds;

	double sum = 0.0;

	for (int i = 0; i < N_prior; ++i) {
		for (int j = 0; j < N_post; ++j) {
			sum += VelocityConsistencyEstimatorHelper::kernel(
				priorSpeeds[i],
				posteriorSpeeds[j],
				sigma2
			);
		}
	}

	return sum / static_cast<double>(N_prior * N_post);
}

LtrResults VelocityConsistencyEstimator::computeTrackLikelihood(
	int N,
	const Eigen::VectorXd& mean_post,
	const Eigen::MatrixXd& cov_post,
	const Eigen::VectorXd& mean_prior,
	const Eigen::MatrixXd& cov_prior)
{
	Eigen::LLT<Eigen::MatrixXd> llt_post(cov_post);
	if (llt_post.info() != Eigen::Success) {
		throw std::runtime_error("Cholesky decomposition failed: the covariance matrix 'cov_post' is not positive definite.");
	}

	Eigen::LLT<Eigen::MatrixXd> llt_prior(cov_prior);
	if (llt_prior.info() != Eigen::Success) {
		throw std::runtime_error("Cholesky decomposition failed: the covariance matrix 'cov_prior' is not positive definite.");
	}

	Eigen::MatrixXd samples_post = sampleFromGaussianLlt(N, mean_post, llt_post);
	Eigen::MatrixXd samples_prior = sampleFromGaussianLlt(N, mean_prior, llt_prior);

	auto speedPriorStats = VelocityConsistencyEstimatorHelper::estimateSpeedStats(velocitySelector, mean_prior, cov_prior);
	meanPriorSpeed = speedPriorStats.first;
	sigmaPriorSpeed = speedPriorStats.second;

	auto speedPostStats = VelocityConsistencyEstimatorHelper::estimateSpeedStats(velocitySelector, mean_post, cov_post);
	meanPostSpeed = speedPostStats.first;
	sigmaPostSpeed = speedPostStats.second;

	if (sigmaPriorSpeed < EstimatorConstants::kMinSigmaPriorSpeed) {
		throw std::runtime_error("sigmaPriorSpeed is too small.");
	}

	posteriorWeights = computeGaussianPdfWeights(samples_post, mean_post, llt_post);
	priorWeights = computeGaussianPdfWeights(samples_prior, mean_prior, llt_prior);

	posteriorSpeeds = computeVelocityNorms(samples_post);
	priorSpeeds = computeVelocityNorms(samples_prior);

	bandwidthPriorSpeeds = VelocityConsistencyEstimatorHelper::bandwidthSilverman(priorSpeeds);
	bandwidthPostSpeeds = VelocityConsistencyEstimatorHelper::bandwidthSilverman(posteriorSpeeds);

	combinedStd = VelocityConsistencyEstimatorHelper::computeCombinedStd(priorSpeeds, posteriorSpeeds);

	posteriorSpeedsResampled = resampleSpeedsWithWeights(samples_post, posteriorWeights, N);
	priorSpeedsResampled = resampleSpeedsWithWeights(samples_prior, priorWeights, N);

	double denominator = computeNormalizationDenominator();
	double denominatorKde = computeNormalizationDenominatorKde();
	double denominatorKdeResampled = computeNormalizationDenominatorKdeResampled();
	double denominatorKernelSimilarity = computeKernelSimilarityDenominator();
	double denominatorKernelSimilarityWeighted = computeKernelSimilarityDenominatorWeighted();

	if (denominator < EstimatorConstants::kMinNormalizationConstant) {
		throw std::runtime_error("NormalizationDenominator is too small.");
	}

	if (denominatorKde < EstimatorConstants::kMinNormalizationConstant) {
		throw std::runtime_error("NormalizationDenominatorKde is too small.");
	}

	if (denominatorKdeResampled < EstimatorConstants::kMinNormalizationConstant) {
		throw std::runtime_error("NormalizationDenominatorKdeResampled is too small.");
	}

	if (denominatorKernelSimilarity < EstimatorConstants::kMinNormalizationConstant) {
		throw std::runtime_error("KernelSimilarityDenominator is too small.");
	}

	if (denominatorKernelSimilarityWeighted < EstimatorConstants::kMinNormalizationConstant) {
		throw std::runtime_error("KernelSimilarityDenominatorWeighted is too small.");
	}
	LtrResults ltrResult;

	ltrResult.kde = computeKde()/ denominator;
	ltrResult.kde_weighted = computeKdeWeighted() / denominator;

	ltrResult.gaussian = computeGaussian()/ denominator;
	ltrResult.gaussian_weighted = computeGaussianWeighted() / denominator;

	/*ltrResult.integral = computeIntegralByHistograms(100);
	ltrResult.integral_weighted = computeIntegralByHistogramsWeighted(100);*/

	ltrResult.kde_resampled = computeKdeResampled() / denominatorKdeResampled;

	ltrResult.kernelSimilarity = computeKernelSimilarity()/ denominatorKernelSimilarity;
	ltrResult.kernelSimilarity_weighted = computeKernelSimilarityWeighted() / denominatorKernelSimilarityWeighted;

	return ltrResult;
}

double VelocityConsistencyEstimator::computeIntegralByHistograms(int bins) {
	double v_min = std::min(priorSpeeds.minCoeff(), posteriorSpeeds.minCoeff());
	double v_max = std::max(priorSpeeds.maxCoeff(), posteriorSpeeds.maxCoeff());
	double h = (v_max - v_min) / bins;

	Eigen::VectorXi histPrior = Eigen::VectorXi::Zero(bins);
	Eigen::VectorXi histPost = Eigen::VectorXi::Zero(bins);

	for (int i = 0; i < priorSpeeds.size(); ++i) {
		int bin = std::min(int((priorSpeeds[i] - v_min) / h), bins - 1);
		histPrior[bin]++;
	}

	for (int i = 0; i < posteriorSpeeds.size(); ++i) {
		int bin = std::min(int((posteriorSpeeds[i] - v_min) / h), bins - 1);
		histPost[bin]++;
	}

	Eigen::VectorXd p1 = histPrior.cast<double>() / (priorSpeeds.size() * h);
	Eigen::VectorXd p2 = histPost.cast<double>() / (posteriorSpeeds.size() * h);
	Eigen::VectorXd kdePrior = Eigen::VectorXd::Zero(bins);
	Eigen::VectorXd kdePost = Eigen::VectorXd::Zero(bins);

	for (int i = 0; i < bins; ++i) {
		double center = v_min + (i + 0.5) * h;

		kdePrior[i] = VelocityConsistencyEstimatorHelper::kdeAtPoint(center, priorSpeeds, bandwidthPriorSpeeds);
		kdePost[i] = VelocityConsistencyEstimatorHelper::kdeAtPoint(center, posteriorSpeeds, bandwidthPostSpeeds);
	}

	// For Test
	double normPrior = p1.sum() * h;
	double normPost = p2.sum() * h;
	double normKdePrior = kdePrior.sum() * h;
	double normKdePost = kdePost.sum() * h;

	std::ofstream file("histograms.txt");
	if (!file.is_open()) {
		std::cerr << "Error in open file!\n";
		return -1.0;
	}

	file << "Center\tHistPrior\tHistPosterior\tKDEPrior\tKDEPosterior\n";

	for (int i = 0; i < bins; ++i) {
		double center = v_min + (i + 0.5) * h;
		file << center << '\t'
			<< p1[i] << '\t'
			<< p2[i] << '\t'
			<< kdePrior[i] << '\t'
			<< kdePost[i] << '\n';
	}
	file.close();
	std::cout << "Histograms saved to file histograms.txt\n";

	double integral = (p1.array() * p2.array()).sum() * h;
	double norm1 = std::sqrt((p1.array().square()).sum() * h);
	double norm2 = std::sqrt((p2.array().square()).sum() * h);
	double normedIntegral = integral / (norm1 * norm2);

	double integral1 = (kdePrior.array() * kdePost.array()).sum() * h;
	norm1 = std::sqrt((kdePrior.array().square()).sum() * h);
	norm2 = std::sqrt((kdePost.array().square()).sum() * h);
	double normedIntegral1 = integral1 / (norm1 * norm2);

	return normedIntegral;
}

double VelocityConsistencyEstimator::computeIntegralByHistogramsWeighted(int bins) {
	double v_min = std::min(priorSpeeds.minCoeff(), posteriorSpeeds.minCoeff());
	double v_max = std::max(priorSpeeds.maxCoeff(), posteriorSpeeds.maxCoeff());
	double h = (v_max - v_min) / bins;

	Eigen::VectorXd histPrior = Eigen::VectorXd::Zero(bins);
	Eigen::VectorXd histPost = Eigen::VectorXd::Zero(bins);

	for (int i = 0; i < priorSpeeds.size(); ++i) {
		int bin = std::min(int((priorSpeeds[i] - v_min) / h), bins - 1);
		histPrior[bin] += priorWeights[i];
	}

	for (int i = 0; i < posteriorSpeeds.size(); ++i) {
		int bin = std::min(int((posteriorSpeeds[i] - v_min) / h), bins - 1);
		histPost[bin] += posteriorWeights[i];
	}

	Eigen::VectorXd p1 = histPrior / (histPrior.sum() * h);
	Eigen::VectorXd p2 = histPost / (histPost.sum() * h);

	Eigen::VectorXd kdePrior = Eigen::VectorXd::Zero(bins);
	Eigen::VectorXd kdePost = Eigen::VectorXd::Zero(bins);

	for (int i = 0; i < bins; ++i) {
		double center = v_min + (i + 0.5) * h;
		kdePrior[i] = VelocityConsistencyEstimatorHelper::weightedKdeAtPoint(center, priorSpeeds, priorWeights, bandwidthPriorSpeeds);
		kdePost[i] = VelocityConsistencyEstimatorHelper::weightedKdeAtPoint(center, posteriorSpeeds, posteriorWeights, bandwidthPostSpeeds);
	}

	// For Test
	double normPrior = p1.sum() * h;
	double normPost = p2.sum() * h;
	double normKdePrior = kdePrior.sum() * h;
	double normKdePost = kdePost.sum() * h;

	// Вывод в файл
	std::ofstream file("histogramsWeighted.txt");
	if (file.is_open()) {
		file << "CenterW\tHistPriorW\tHistPosteriorW\tKDEPriorW\tKDEPosteriorW\n";
		for (int i = 0; i < bins; ++i) {
			double center = v_min + (i + 0.5) * h;
			file << center << '\t'
				<< p1[i] << '\t'
				<< p2[i] << '\t'
				<< kdePrior[i] << '\t'
				<< kdePost[i] << '\n';
		}
		file.close();
		std::cout << "Histograms saved to file histogramsWeighted.txt\n";
	}
	else {
		std::cerr << "Error opening histogramsWeighted.txt\n";
		return -1.0;
	}

	double integral = (p1.array() * p2.array()).sum() * h;
	double norm1 = std::sqrt((p1.array().square()).sum() * h);
	double norm2 = std::sqrt((p2.array().square()).sum() * h);
	double normedIntegral = integral / (norm1 * norm2);

	double integral1 = (kdePrior.array() * kdePost.array()).sum() * h;
	norm1 = std::sqrt((kdePrior.array().square()).sum() * h);
	norm2 = std::sqrt((kdePost.array().square()).sum() * h);
	double normedIntegral1 = integral1 / (norm1 * norm2);

	return normedIntegral;
}


Eigen::VectorXd VelocityConsistencyEstimator::resampleSpeedsWithWeights(
	const Eigen::MatrixXd& samples,
	const Eigen::VectorXd& weights,
	int N)
{
	if (samples.rows() != weights.size()) {
		throw std::runtime_error("samples and weights size mismatch");
	}

	std::discrete_distribution<> dist(weights.data(), weights.data() + weights.size());

	Eigen::MatrixXd resampledSamples(N, samples.cols());
	for (int i = 0; i < N; ++i) {
		int idx = dist(randomGenerator);
		resampledSamples.row(i) = samples.row(idx);
	}

	return computeVelocityNorms(resampledSamples);
}


double VelocityConsistencyEstimator::computeKdeResampled() {
	if (bandwidthPriorSpeeds < 0) {
		throw std::runtime_error("Bandwidth not initialized");
	}

	double L_tr = 0.0;
	for (int i = 0; i < posteriorSpeedsResampled.size(); ++i) {
		L_tr += VelocityConsistencyEstimatorHelper::kdeAtPoint(
			posteriorSpeedsResampled(i), priorSpeeds, bandwidthPriorSpeeds);
	}

	return L_tr / static_cast<double>(posteriorSpeedsResampled.size());
}

double VelocityConsistencyEstimator::computeNormalizationDenominatorKdeResampled() {
	if (bandwidthPriorSpeeds < 0)  throw std::runtime_error("bandwidthPriorSpeeds not initialized");
	if (bandwidthPostSpeeds  < 0)  throw std::runtime_error("bandwidthPostSpeeds not initialized");

	double sum_post = 0.0;
	for (int i = 0; i < posteriorSpeedsResampled.size(); ++i)
		sum_post += VelocityConsistencyEstimatorHelper::kdeAtPoint(posteriorSpeedsResampled(i), posteriorSpeeds, bandwidthPostSpeeds);
	sum_post /= static_cast<double>(posteriorSpeedsResampled.size());

	double sum_prior = 0.0;
	for (int i = 0; i < priorSpeedsResampled.size(); ++i)
		sum_prior += VelocityConsistencyEstimatorHelper::kdeAtPoint(priorSpeedsResampled(i), priorSpeeds, bandwidthPriorSpeeds);
	sum_prior /= static_cast<double>(priorSpeedsResampled.size());

	return std::sqrt(sum_post * sum_prior);
}

double VelocityConsistencyEstimator::computeKdeInnerProductDenominator() {
	if (bandwidthPriorSpeeds <= 0.0 || bandwidthPostSpeeds <= 0.0) {
		throw std::runtime_error("Bandwidths not initialized");
	}

	const int N_prior = static_cast<int>(priorSpeeds.size());
	const int N_post = static_cast<int>(posteriorSpeeds.size());

	if (N_prior == 0 || N_post == 0) {
		throw std::runtime_error("Empty speed arrays");
	}

	double sigma2_prior = 2.0 * bandwidthPriorSpeeds * bandwidthPriorSpeeds;
	double sum_prior = 0.0;

	for (int i = 0; i < N_prior; ++i) {
		for (int j = 0; j < N_prior; ++j) {
			sum_prior += VelocityConsistencyEstimatorHelper::kernel(
				priorSpeeds[i],
				priorSpeeds[j],
				sigma2_prior
			);
		}
	}

	double L_prior = sum_prior / (N_prior * N_prior);

	double sigma2_post = 2.0 * bandwidthPostSpeeds * bandwidthPostSpeeds;
	double sum_post = 0.0;

	for (int i = 0; i < N_post; ++i) {
		for (int j = 0; j < N_post; ++j) {
			sum_post += VelocityConsistencyEstimatorHelper::kernel(
				posteriorSpeeds[i],
				posteriorSpeeds[j],
				sigma2_post
			);
		}
	}

	double L_post = sum_post / (N_post * N_post);

	return std::sqrt(L_prior * L_post);
}

double VelocityConsistencyEstimator::computeKdeInnerProductDenominatorWeighted() {
	if (bandwidthPriorSpeeds <= 0.0 || bandwidthPostSpeeds <= 0.0) {
		throw std::runtime_error("Bandwidths not initialized");
	}

	if (posteriorSpeeds.size() == 0 || priorSpeeds.size() == 0) {
		throw std::runtime_error("Empty speed arrays");
	}

	if (posteriorWeights.size() == 0 || priorWeights.size() == 0) {
		throw std::runtime_error("Empty weights arrays");
	}

	if (posteriorSpeeds.size() != posteriorWeights.size()) {
		throw std::runtime_error("posteriorSpeeds and posteriorWeights must have the same size.");
	}

	if (priorSpeeds.size() != priorWeights.size()) {
		throw std::runtime_error("priorSpeeds and priorWeights must have the same size.");
	}

	const int N_prior = static_cast<int>(priorSpeeds.size());
	const int N_post = static_cast<int>(posteriorSpeeds.size());


	double sigma2_prior = 2.0 * bandwidthPriorSpeeds * bandwidthPriorSpeeds;

	double sum_prior = 0.0;

	for (int i = 0; i < N_prior; ++i) {
		for (int j = 0; j < N_prior; ++j) {
			double w = priorWeights[i] * priorWeights[j];

			sum_prior += w * VelocityConsistencyEstimatorHelper::kernel(
				priorSpeeds[i],
				priorSpeeds[j],
				sigma2_prior
			);
		}
	}

	double sigma2_post = 2.0 * bandwidthPostSpeeds * bandwidthPostSpeeds;
	double sum_post = 0.0;

	for (int i = 0; i < N_post; ++i) {
		for (int j = 0; j < N_post; ++j) {
			double w = posteriorWeights[i] * posteriorWeights[j];

			sum_post += w * VelocityConsistencyEstimatorHelper::kernel(
				posteriorSpeeds[i],
				posteriorSpeeds[j],
				sigma2_post
			);
		}
	}

	return std::sqrt(sum_prior * sum_post);
}
