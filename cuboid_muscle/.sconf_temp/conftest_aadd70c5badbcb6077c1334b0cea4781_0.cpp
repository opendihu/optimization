

#include <Python.h>  // this has to be the first included header

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char* argv[])
{
  std::string pythonSearchPath = "/home/lukas/Desktop/Bachelorthesis/OpenDiHu/opendihu/dependencies/python/install";
  
  std::cout << "pythonSearchPath: [" << pythonSearchPath << "]" << std::endl;
  
  //std::string pythonSearchPath = std::string("/store/software/opendihu/dependencies/python/install");
  const wchar_t *pythonSearchPathWChar = Py_DecodeLocale(pythonSearchPath.c_str(), NULL);
  Py_SetPythonHome((wchar_t *)pythonSearchPathWChar);

  Py_Initialize();
  
  PyEval_InitThreads();
  Py_SetStandardStreamEncoding(NULL, NULL);

  // check if numpy module could be loaded
  PyObject *numpyModule = PyImport_ImportModule("numpy");
  if (numpyModule == NULL)
  {
    std::cout << "Failed to import numpy." << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}

