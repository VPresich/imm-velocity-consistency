#pragma once
#include <iostream>
#include <vector>
#include "PythonBridge.h"
#include "EnvLoader.h"


void runPlotPython(const std::vector<double>& x, const std::vector<double>& kde_post,
	const std::vector<double>& kde_prior, const std::vector<double>& gausian_prior, double v_min, double v_max, double kernel_similarity,
	const std::string& save_path)
{
	EnvLoader::load("config.env");
	try
	{
        PythonBridge py;

		std::vector<PyObject*> args;
		args.push_back(py.to_pyobject(x));
		args.push_back(py.to_pyobject(kde_post));
		args.push_back(py.to_pyobject(kde_prior));
		args.push_back(py.to_pyobject(gausian_prior));
		args.push_back(py.to_pyobject(v_min));
		args.push_back(py.to_pyobject(v_max));

		if (kernel_similarity >= 0.0)
			args.push_back(py.to_pyobject(kernel_similarity));
		else {
			Py_INCREF(Py_None);
			args.push_back(Py_None);
		}

		if (!save_path.empty())
			args.push_back(py.to_pyobject(save_path));
		else {
			Py_INCREF(Py_None);
			args.push_back(Py_None);
		}

		PyObject* result = py.call_function("velocity_diagnostic", "plot_graph", args);
		if (!result)
		{
			PyErr_Print();
			std::cerr << "Error calling function 'plot_graph'" << std::endl;
		}

		Py_XDECREF(result);
		for (auto obj : args) Py_XDECREF(obj);
	}
	catch (const std::exception& e)
	{
		std::cerr << "PythonBridge error: " << e.what() << std::endl;
	}
}