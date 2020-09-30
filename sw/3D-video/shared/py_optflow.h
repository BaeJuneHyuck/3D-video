#pragma once

#include <stdlib.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <opencv2/opencv.hpp>
#include <string>

#include <filesystem>

using namespace std;
using namespace cv;

// < config below >---

const string module_name{ "helper" };
const string function_name{ "run" };

// ---< config above >

PyObject* py_func;

void py_calc_optflow(string path1, string path2, Mat& flow)
{
	PyObject* args = PyTuple_New(2);
	PyObject* string1 = PyUnicode_FromString(path1.c_str());
	PyObject* string2 = PyUnicode_FromString(path2.c_str());
	PyTuple_SetItem(args, 0, string1);
	PyTuple_SetItem(args, 1, string2);

	PyObject* ret_obj = PyObject_CallObject(py_func, args);

	Py_DECREF(args);

	PyArrayObject* ret_array = reinterpret_cast<PyArrayObject*>(ret_obj);
	ret_array = PyArray_GETCONTIGUOUS(ret_array);

	int rows = PyArray_SHAPE(ret_array)[0], cols = PyArray_SHAPE(ret_array)[1];
	float* p;

	Mat ret(rows, cols, CV_32FC2);

	p = reinterpret_cast<float*>(PyArray_DATA(ret_array));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			ret.at<Point2f>(i, j) = Point2f(p[i * cols * 2 + j * 2], p[i * cols * 2 + j * 2 + 1]);
		}
	}

	Py_DECREF(ret_array);

	flow = ret;
}

void py_initialize()
{
	filesystem::current_path("./RAFT/");

	Py_Initialize();
	_import_array();
	PyRun_SimpleString("import sys; sys.path.insert(0, '.')");
	PyObject* py_module_name = PyUnicode_FromString(module_name.c_str());
	PyObject* module = PyImport_Import(py_module_name);
	Py_DECREF(py_module_name);
		
	py_func = PyObject_GetAttrString(module, function_name.c_str());
}