/*
*  This file exists to be compiled to verify if headers are found
*   if not we do not add the -D arguments for MKL
*/

#include <mkl.h>

void  PyInit_blas(){}
