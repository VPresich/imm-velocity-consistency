#pragma once
#include <Python.h>
#include <string>
#include <vector>

class PythonBridge
{
public:
	PythonBridge(const std::string& pythonPath = "");
	~PythonBridge();

	PyObject* call_function(const std::string& moduleName,
		const std::string& funcName,
		const std::vector<PyObject*>& args);

	PyObject* to_pyobject(int x);
	PyObject* to_pyobject(double x);
	PyObject* to_pyobject(const std::string& s);
	PyObject* to_pyobject(const std::vector<int>& v);
	PyObject* to_pyobject(const std::vector<double>& v);
};
