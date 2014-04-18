Cuda_MatrixMultiplication
=========================

Cuda code for Square Matrix Multiplication 

RUN IN CMD LINE: nvcc -o [outputfilename] [filename] will generate a .exe, .lib and .exp files

EXAMPLE:
nvcc -o sample1 cuda_matrmul.cu 
Generates sample1.exe, run .exe to get

OUTPUT: 
Executing Matrix Multiplication;
Matrix size : 1000x1000;
Time taken is : 15.817;
Finished.

MATRIX SIZE: Change the value in "#define BLOCK_SIZE 10" for different matrix_sizes. Matrix size = BLOCK_SIZE*100;
