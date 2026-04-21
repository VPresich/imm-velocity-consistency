#pragma once
#include <vector>
#include "VelocityExperiment.h"

void runVelocityEstimatorDemo(VelocityExperiment experiment = VelocityExperiment::Example0);
void runPlotPython(const std::vector<double>& x, const std::vector<double>& kde_post,
	const std::vector<double>& kde_prior, const std::vector<double>& gausian_prior, double v_min, double v_max, double kernel_similarity = -1.0,
	const std::string& save_path = "graph.png");