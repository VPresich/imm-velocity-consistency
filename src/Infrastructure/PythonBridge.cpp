#include "PythonBridge.h"
#include <iostream>
#include <stdexcept>

#define WIDEN2(x) L ## x
#define WIDEN(x) WIDEN2(x)

PythonBridge::PythonBridge(const std::string& pythonPath)
{
    Py_Initialize();
	if (!Py_IsInitialized())
		throw std::runtime_error("Python initialization failed!");

	if (!pythonPath.empty()) {
		PyObject* sysPath = PySys_GetObject("path");
		PyList_Append(sysPath, PyUnicode_FromString(pythonPath.c_str()));
	}
}


PythonBridge::~PythonBridge()
{
	if (Py_IsInitialized())
		Py_Finalize();
}

PyObject* PythonBridge::call_function(const std::string& moduleName,
	const std::string& funcName,
	const std::vector<PyObject*>& args)
{
	PyObject* pModule = PyImport_ImportModule(moduleName.c_str());
	if (!pModule) throw std::runtime_error("Module '" + moduleName + "' not found");

	PyObject* pFunc = PyObject_GetAttrString(pModule, funcName.c_str());
	if (!pFunc || !PyCallable_Check(pFunc)) {
		Py_DECREF(pModule);
		throw std::runtime_error("Function '" + funcName + "' not found or not callable");
	}

	PyObject* argsTuple = PyTuple_New(args.size());
	for (size_t i = 0; i < args.size(); i++)
		PyTuple_SetItem(argsTuple, i, args[i]);

	PyObject* result = PyObject_CallObject(pFunc, argsTuple);
	Py_DECREF(argsTuple);
	Py_DECREF(pFunc);
	Py_DECREF(pModule);

	if (!result) throw std::runtime_error("Error calling function '" + funcName + "'");
	return result;
}

PyObject* PythonBridge::to_pyobject(int x) { return PyLong_FromLong(x); }
PyObject* PythonBridge::to_pyobject(double x) { return PyFloat_FromDouble(x); }
PyObject* PythonBridge::to_pyobject(const std::string& s) { return PyUnicode_FromString(s.c_str()); }

PyObject* PythonBridge::to_pyobject(const std::vector<int>& v) {
	PyObject* list = PyList_New(v.size());
	for (size_t i = 0; i < v.size(); ++i)
		PyList_SetItem(list, i, PyLong_FromLong(v[i]));
	return list;
}

PyObject* PythonBridge::to_pyobject(const std::vector<double>& v) {
	PyObject* list = PyList_New(v.size());
	for (size_t i = 0; i < v.size(); ++i)
		PyList_SetItem(list, i, PyFloat_FromDouble(v[i]));
	return list;
}
