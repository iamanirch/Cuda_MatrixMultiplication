Cuda_MatrixMultiplication
=========================

Cuda code for Square Matrix Multiplication 

RUN IN CMD LINE: nvcc -o [outputfilename] [filename] will generate a .exe, .lib and .exp files

EXAMPLE:
nvcc -o sample1 cuda_matrmul.cu 
Generates sample1.exe, run .exe to get

SAMPLE OUTPUT: 
Executing Matrix Multiplication;
Matrix size : 100x100;
Time taken is : .100;
Finished.
Matrix size : 200x200;
Time taken is : 0.140;
Finished.
Matrix size : 300x300;
Time taken is : 0.329;
Finished.
Matrix size : 400x400;
Time taken is : 0.688;
Finished.
Matrix size : 500x500;
Time taken is : 1.312;
Finished.
Matrix size : 600x600;
Time taken is : 2.157;
Finished.
Matrix size : 700x700;
Time taken is : 3.484;
Finished.
Matrix size : 800x800;
Time taken is : 7.233;
Finished.
Matrix size : 900x900;
Time taken is : 14.06;
Finished.
Matrix size : 1000x1000;
Time taken is : 16.52;
Finished.
MATRIX SIZE defined as BLOCK_SIZE * 100. The entire program runs in a loop where BLOCK_SIZE goes from 1 to 10.
