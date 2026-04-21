#pragma once
#include <Eigen/Dense>
#include <random>

struct LtrResults {
	double kde;
	double kde_weighted;
	double gaussian;
	double gaussian_weighted;
	double integral;
	double integral_weighted;
	double kde_resampled;
	double kernelSimilarity;
	double kernelSimilarity_weighted;
};

class VelocityConsistencyEstimator {
public:
	VelocityConsistencyEstimator(int vel_dim, int st_dim, int vel_offset);

	LtrResults computeTrackLikelihood(int N, const Eigen::VectorXd& mean_post,
		const Eigen::MatrixXd& cov_post, const Eigen::VectorXd& mean_prior, const Eigen::MatrixXd& cov_prior);

	const Eigen::VectorXd& getPriorSpeeds() const { return priorSpeeds; }
	const Eigen::VectorXd& getPosteriorSpeeds() const { return posteriorSpeeds; }
	const Eigen::MatrixXd& getVelocitySelector() const { return velocitySelector; }

private:
	std::mt19937 randomGenerator;
	std::normal_distribution<> normalDistribution;
	int velocityDim;
	int stateDim;
	int velocityOffset;

	Eigen::MatrixXd velocitySelector;
	Eigen::VectorXd priorSpeeds;
	Eigen::VectorXd posteriorSpeeds;

	Eigen::VectorXd posteriorWeights;
	Eigen::VectorXd priorWeights;

	Eigen::VectorXd posteriorSpeedsResampled;
	Eigen::VectorXd priorSpeedsResampled;

	double meanPriorSpeed = -1;
	double sigmaPriorSpeed = -1;
	double bandwidthPriorSpeeds = -1;

	double meanPostSpeed = -1;
	double sigmaPostSpeed = -1;
	double bandwidthPostSpeeds = -1;

	double combinedStd = -1;

	void initializeVelocitySelector();
	Eigen::MatrixXd sampleFromGaussianLlt(int N, const Eigen::VectorXd& mean, const Eigen::LLT<Eigen::MatrixXd>& llt);

	Eigen::VectorXd computeGaussianPdfWeights(const Eigen::MatrixXd& samples, const Eigen::VectorXd& mean, const Eigen::LLT<Eigen::MatrixXd>& llt);
	Eigen::VectorXd computeVelocityNorms(const Eigen::MatrixXd& samples);
	Eigen::VectorXd resampleSpeedsWithWeights(const Eigen::MatrixXd& samples, const Eigen::VectorXd& weights, int N);

	double computeKde();
	double computeGaussian();
	double computeKdeWeighted();
	double computeGaussianWeighted();
	double computeIntegralByHistograms(int bins);
	double computeIntegralByHistogramsWeighted(int bins);
	double computeKernelSimilarity();
	double computeKernelSimilarityWeighted();
	double computeKdeInnerProduct();
	double computeKdeInnerProductWeighted();

	double computeNormalizationDenominator();
	double computeNormalizationDenominatorKde();
	double computeKdeResampled();
	double computeNormalizationDenominatorKdeResampled();
	double computeKernelSimilarityDenominator();
	double computeKernelSimilarityDenominatorWeighted();
	double computeKdeInnerProductDenominator();
	double computeKdeInnerProductDenominatorWeighted();
};
