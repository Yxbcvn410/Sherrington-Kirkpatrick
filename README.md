# Sherrington-Kirkpatrick_cpp

This program can solve a given Sherrington-Kirkpatrick problem using a Simulated Annealing algorithm. It uses CUDA to perform hardware-accelerated calculations.

# How to asssemble

Clone repository and run 'make all' from directory 'Release'. Makefiles are provided.

# How to use

To run the program, you should first create a file called 'config' in its working directory and define parameters in a way that is described below.
You can also pass all parameter definitions as arguments.  
Please note that this program is supposed to be run on a Unix-like system. You should also have 'gnuplot' installed if you want to use integrated plotting functions (gnuplot script is generated anyway)

# Config file syntaxis

\# to comment a line  
% to define a parameter value

*Parameter list:*  
%start - Temperature start value, Default is 0.  
%end or %e - Temperature stop value  
%step or %s - Temperature step value, overrides %count  
%count or %c - Temperature point quantity, overrides %step. Default is 1000  
%pstep or %ps - Annealing step value. Default is 1
%mindiff or %md - Spin difference threshold value (Algorithm proceeds to next step if diff is less than %md). Default is 0.01  
%msize or %ms - Matrice size. If this parameter is defined, a new matrice with random numbers and given size will be created  
%mloc or %ml - Full path to matrice file  
%mloc_b or %ml_b - Full path to graph file  
%wdir or %wd - Output directory path. Set this parameter to "-a" to create it automatically in the program's working directory
%initrand or %ird - Set this to true if you want to initialize random generator, false otherwise. Default is false  
%bcount or %bc - CUDA block count  
%appconf or %ac - set this to "both" or "b" if you want to append best run upper and lower temperature bounds to the config file after all calculations, "upper" or "u" to append upper bound only, and "none" or "n" to disable bounds appending  
%cli - set this to true if you want to enable block progress view, false otherwise. Default is false  
%plot - set this to true to enable plotting, false otherwise. Default is false  
%lincoef or %lc - Annealing linear coefficent. Default is 1