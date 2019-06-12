################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/CudaOperator.cu 

CPP_SRCS += \
../src/FilesystemProvider.cpp \
../src/Matrix.cpp \
../src/Plotter.cpp \
../src/Spinset.cpp \
../src/StartupUtils.cpp \
../src/main.cpp 

OBJS += \
./src/CudaOperator.o \
./src/FilesystemProvider.o \
./src/Matrix.o \
./src/Plotter.o \
./src/Spinset.o \
./src/StartupUtils.o \
./src/main.o 

CU_DEPS += \
./src/CudaOperator.d 

CPP_DEPS += \
./src/FilesystemProvider.d \
./src/Matrix.d \
./src/Plotter.d \
./src/Spinset.d \
./src/StartupUtils.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -O3 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc -O3 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -O3 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


