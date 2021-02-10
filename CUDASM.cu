///============================================================================
// Name  : CUDASM.cu
// Author      : Or Alus
// Version     :
// Copyright   :  This is a adaptaion to cuda of another program used to produce some of the results in https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.032204 if you use it, 
//		  Please cite that paper or as Alus Or, StandradMap, 2020 Github reposetory,  https://github.com/orralus/StandardMap.
// Description : This is a cuda implementation to observe superdiffusion and survival probality in the presence of accelerator mode island in the standard map
//		 The script was written in my Ph.D at the Physics Dept. at the Technion.
//		 It  takes N initial condition at (x = [-0.01, 0.01], y=0) and uses the standard map (x', y') = (x + y + K sin(x), y + K sin(x)) on a torus to propegate to time T. 
// 		 It gives an estimate for the diffusion coeficient and the survival probability (the probablitiy to stay on one side of the torus).
//============================================================================
#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>
#include <ctime>
#include <sstream>
#include <fstream>
#include <limits.h>
#include <iomanip>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    fprintf(stderr,"Error %s at %s:%d\n",cudaGetErrorString(x),__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void reduce0( double *g_idata, double *g_odata) 
{
	extern __shared__ double sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x
	*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void setup_kernel( unsigned long long seed, curandState *state)
{ 
    //int id = threadIdx.x + blockIdx.x * blockDim.x ; //number of states=N

      int tid=threadIdx.x; //number of states is number of threads 
    /* Each thread gets same seed, a different sequence 
       number, no offset */ 
	curand_init(seed, tid, 0, &state[tid]);

}
__global__ void set_values_kernel( double *ioData, double value, unsigned int length)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x ;
    /* Each thread gets same seed, a different sequence 
       number, no offset */ 
	if (id < length){
		ioData[id] = value;
	}
}
__global__ void set_values_kernel( int *ioData, int value, unsigned int length)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x ;
    /* Each thread gets same seed, a different sequence 
       number, no offset */ 
    if (id < length){
		ioData[id] = value;
	}
}

__global__ void set_values_kernel( unsigned long long *ioData, unsigned long long  value, unsigned int length) 
{
    int id = threadIdx.x + blockIdx.x * blockDim.x ;
    /* Each thread gets same seed, a different sequence 
       number, no offset */ 
   if (id<length){
   ioData[id]=value;
  }
}
__global__ void generate_randomwalk_kernel( curandState *state, int n, double *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int tid= threadIdx.x;
    double x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[tid]; //number of states=threadsPerBlock
    /* Generate pseudo-random uniforms */
    for(int i = 0; i < n; i++) {
        x = curand_uniform( &localState);
        /* store result */
        if(x > .5) {
            result[id] = 1.0;
        } else {
	    result[id] = -1.0;	
		}
	}
    /* Copy state back to global memory */
    state[tid] = localState; //number of states=threadsPerBlock
}

__global__ void generate_uniform_kernel(curandState *state, int n, double* result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    double x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[tid];//number of states=threadsPerBlock
    /* Generate pseudo-random uniforms */
    for(int i = 0; i < n; i++) {
        x = curand_uniform(&localState);      
    }
    /* Copy state back to global memory */
    state[tid] = localState;//number of states=threadsPerBlock
   /* store result */
   result[id] = 0.02*x-0.01;
}

 __global__ void MapAdvance (double *X, double *Y, double *k, double *Xsqr, int *tcounter, unsigned long long *starters, unsigned long long *P)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
	int temp;
    double x0 = X[id]; 
	double y0 =Y[id];
	Y[id] = y0 - (*k) * sin(x0); //map step
	X[id] = x0 + Y[id];
	if (sin(X[id]) * sin(x0) > 0){ //check for passing from one side of the torus to the other
		tcounter[id] += 1;
	} else {								//update survival probability
		temp = tcounter[id]; 
		tcounter[id] = 0;
		atomicAdd(P + temp, 1);
		atomicAdd(starters, 1);
	}
	Xsqr[id] = powf(X[id], 2);	//calculate variance
 }
 

int main(int argc, char** argv){

    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    int N;
    N = 1024*1024*8;	//Number of initial conditions
    int T = 10000000; 	//Number of time steps
   
	int threadsPerBlock=1024;
    int numBlocks = N / threadsPerBlock; 
    size_t size = N * sizeof(double);
    unsigned long long *P;  	// Survival probability
	double *D; 	//Diffusion coeficient
   
   if (!(D = new double[T])){
        std::cout<<"Error: Out of memory."<<std::endl;
        return  EXIT_FAILURE;
    }
    if (!(P = new unsigned long long[T])){
        std::cout<<"Error: Out of memory."<<std::endl;
        return  EXIT_FAILURE;
    }
    int *tcounter;
    if (!(tcounter = new int[N])){
        std::cout<<"Error: Out of memory."<<std::endl;
        return  EXIT_FAILURE;
    }
    cout<<"INT_MAX="<<INT_MAX<<"\n";

    time_t tstart, tend;
    int ii;

	//declaring varibales on GPU
    double* vec;
    CUDA_CALL(cudaMalloc((void**)&vec, size));
    
    double* X;
    CUDA_CALL(cudaMalloc((void**)&X, size));
   
    double* Y;
    CUDA_CALL(cudaMalloc((void**)&Y, size));
    
	double* Xsqr;
    CUDA_CALL(cudaMalloc((void**)&Xsqr, size));

   unsigned long long* devP;
    CUDA_CALL(cudaMalloc((void**)&devP, sizeof(unsigned long long)*T));
    
	unsigned long long* devstarters;
    CUDA_CALL(cudaMalloc((void**)&devstarters, sizeof(unsigned long long)));
    
	double* devK;
    CUDA_CALL(cudaMalloc((void**)&devK, sizeof(double)));
    
	double* devTemp;
    CUDA_CALL(cudaMalloc((void**)&devTemp, size));
    
	double* devTempnext;
     CUDA_CALL(cudaMalloc((void**)&devTempnext,size));
    
    int* devtcounter;
    CUDA_CALL(cudaMalloc((void**)&devtcounter, sizeof(int)*N));
     
    double* Temp;

  
    //initialize random number generator
    
     curandState *devStates;
     CUDA_CALL(cudaMalloc((void **)&devStates, threadsPerBlock * sizeof(curandState))); // number pf states = threadsPerBlock
     setup_kernel<<< 1, threadsPerBlock>>>(seed, devStates);

     //initialize varibales on device
    generate_uniform_kernel<<<numBlocks,threadsPerBlock>>>(devStates,1,X);   //generate values between -0.01..0.01
    
    set_values_kernel<<<numBlocks,threadsPerBlock>>>(devtcounter ,0.0,N);  
    set_values_kernel<<<numBlocks,threadsPerBlock>>>(Y, 0.0,N);  
    set_values_kernel<<<numBlocks,threadsPerBlock>>>(devP, 0,T);
    set_values_kernel<<<numBlocks,threadsPerBlock>>>(devK, stod(argv[2]), 1);
    set_values_kernel<<<numBlocks,threadsPerBlock>>>(devstarters, N, 1);
	
    ofstream myfile;
    myfile.open ("Test.csv");
    
	ofstream myfile2;
    myfile2.open(argv[1]);
    
	tstart = time(0);
    
	unsigned long long starters=0;
    
	for (int t = 0; t < T; t++) { //iterate through time
        D[t] = 0.0;
		
       	MapAdvance <<<numBlocks,threadsPerBlock>>>  (X,Y, devK, Xsqr, devtcounter, devstarters,devP); //propegate the map
        
		int Blocks=numBlocks;
        int TempN=N;
		
        //Use reduce sum to calculate <x^2>
		
		set_values_kernel<<<numBlocks, threadsPerBlock>>>(devTemp, 0.0, N);
        set_values_kernel<<<numBlocks, threadsPerBlock>>>(devTempnext, 0.0, N);
     	reduce0 <<<Blocks, threadsPerBlock, threadsPerBlock * sizeof(double) >>>(Xsqr, devTemp); 
		
		while (Blocks>=threadsPerBlock) {       
			TempN = Blocks;
			Blocks = Blocks/threadsPerBlock;
			reduce0 <<<Blocks, threadsPerBlock, threadsPerBlock*sizeof(double) >>>(devTemp, devTempnext);
			set_values_kernel<<<numBlocks, threadsPerBlock>>>(devTemp, 0.0, TempN);
			CUDA_CALL(cudaMemcpy(devTemp, devTempnext, Blocks*sizeof(double), cudaMemcpyDeviceToDevice));
			set_values_kernel<<<numBlocks, threadsPerBlock>>>(devTempnext, 0.0, Blocks);
       	}

        if( !(Temp  = new double[Blocks] ))
        {
           cout << "Error: out of memory." <<endl;
           exit(1);
        }
		
        CUDA_CALL(cudaMemcpy(Temp,devTemp,Blocks*sizeof(double),cudaMemcpyDeviceToHost)); 
        
		for (ii=0; ii<Blocks; ii++) {
			*(D+t) +=(double)Temp[ii]/N;
		} 
        
		delete [] Temp;
     }
  

	CUDA_CALL(cudaMemcpy(P,devP,sizeof(unsigned long long)*T,cudaMemcpyDeviceToHost)); 
    CUDA_CALL(cudaMemcpy(tcounter,devtcounter,sizeof(int)*N,cudaMemcpyDeviceToHost)); 
    CUDA_CALL(cudaMemcpy(&starters,devstarters,sizeof(unsigned long long),cudaMemcpyDeviceToHost)); 	
    
	CUDA_CALL(cudaFree(X));
    CUDA_CALL(cudaFree(Xsqr));
    CUDA_CALL(cudaFree(devP));
    CUDA_CALL(cudaFree(devstarters));
    CUDA_CALL(cudaFree(devK));
    CUDA_CALL(cudaFree(devtcounter));
    CUDA_CALL(cudaFree(devStates));
    CUDA_CALL(cudaFree(devTemp));
    
	long long N0;
    N0 = starters;
    long long sum = 0;
    for (ii = 0; ii < T; ii++) {
        sum += (*(P + ii));
        myfile2<< setprecision(15) << (double) (N0 - sum) / N0 << ", ";
     }
	myfile2<< "\n";
    
	for (int t=0; t<T;t++) {myfile2<< *(tcounter+t)<<",";}
    
	myfile2<<std::endl;
    
	for (int t=0; t<T;t++) {myfile2<< *(D+t)<<",";}
    
	myfile2<<std::endl;
    
	for (int t=0; t<T;t++) {myfile2<< *(P+t)<<",";}
    
	myfile2<<std::endl;
	myfile2 << N0 << "\n";
	
    tend = time(0);
      
    std::cout << "It took " << difftime(tend, tstart) << " second(s)." << std::endl;
    
	myfile.close();
    myfile2.close();

    return 0;
}
