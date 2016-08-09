#NVCC := /usr/local/cuda/bin/nvcc
NVCC := /usr/local/cuda/bin/nvcc -I/usr/local/cuda/include
#NVCC := nvcc
CC   := g++ -I/usr/local/cuda/include

all:BPtrain clean


BPtrain: BPtrain.cc BP_GPU.o DevFunc.o Interface.o
	${NVCC}  BPtrain.cc BP_GPU.o DevFunc.o Interface.o -o BPtrain -L/usr/local/cuda/lib64 -lcublas -lcurand	
Interface.o: Interface.h Interface.cc
	${CC} -c Interface.cc
DevFunc.o: DevFunc.h DevFunc.cu
	${NVCC} -c DevFunc.cu
BP_GPU.o: BP_GPU.h BP_GPU.cu DevFunc.o
	${NVCC} -c BP_GPU.cu



clean:
		rm DevFunc.o BP_GPU.o Interface.o