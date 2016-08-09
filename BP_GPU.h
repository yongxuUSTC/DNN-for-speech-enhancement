//#include "/usr/local/cuda/include/cuda_runtime.h"
#include <cuda_runtime.h>
//#include "/usr/local/cuda/include/cublas_v2.h"
#include <cublas_v2.h>
//#include "/usr/local/cuda-5.0/include/cuda_runtime.h"
//#include "/usr/local/cuda-5.0/include/cublas_v2.h"
//#include "/usr/local/cuda-5.0/include/curand.h" //这里可能提示cuda_runtime.h找不到，就要进入curand.h，把cuda_runtime.h的全路径贴上去
#include <curand.h>
//#include "/usr/local/cuda/include/cuda_runtime.h"
//#include "/usr/local/cuda/include/cublas_v2.h"
//#include "/usr/local/cuda/include/curand.h"

#define MAXLAYER	10
#define MAXCACHEFRAME 200000                                

struct BP_WorkSpace
{
	float *in; 								/// input data
	float *out;								/// Output data，输出是float型，它通过softmax变成0,1
	//int *targ;							  /// target data，我要改#############################，target是int，得改成float
    float *targ;////////////////////////////////////by yongxu

	float *weights[MAXLAYER];   	/// weights for layers，指针数组，每个数组都是指针，本质是数组，权重是二维的；；int (*p)[10]; p即为指向数组元素地址的指针，本质是指针
	float *bias[MAXLAYER];      	/// biases for layers；rbm的输入，输出各有一个bias

	float *layer_x[MAXLAYER];  	  /// Input to layer
	float *layer_y[MAXLAYER]; 		/// Output from layer
	float *layer_dedy[MAXLAYER];  /// de/dy
	float *layer_dydx[MAXLAYER];  /// dy/dx
	float *layer_dedx[MAXLAYER];  /// de/dx
	float *layer_ydedx[MAXLAYER];
	float *layer_sumdedx[MAXLAYER]; 

	float *delta_bias[MAXLAYER]; // Output bias update
	float *delta_weights[MAXLAYER]; // Output bias update
		float *DevRandVector; //Dropout随机数存储矩阵
	int *DevSeed;//Dropout随机数种子
};

class BP_GPU
{
public:
	BP_GPU(int a_GPU_selected, int a_numlayers, int *a_layersizes, int a_bunchsize, float a_lrate, float a_momentum, float  a_weightcost,
		float **weights, float **bias,int dropoutflag, float visible_omit,float hid_omit);
	~BP_GPU();
public:
	//void train(int n_frames, const float* in, const int *targ);
	//void train(int n_frames, const float* in, const float *targ);////////////////////////////////////////by yongxu
	void train(int n_frames, float* in, const float *targ);
	//void train_bunch_multi(int n_frames,  float** in, int **targ);
	//void train_bunch_multi(int n_frames,  float** in, float **targ);//////////////////////////////////////by yongxu
	void train_bunch_multi(int n_frames,  float** in, float **targ);
	//void train_bunch_single(int n_frames, const float* in, const int *targ);
	//void train_bunch_single(int n_frames, const float* in, const float *targ);//////////////////////////////by yongxu
	void train_bunch_single(int n_frames, float* in, const float *targ);
	//int  CrossValid(int n_frames, const float* in, const int *targ);	
	float  CrossValid(int n_frames, const float* in, const float *targ);///////////////////////////////////////by yongxu
	//void cv_bunch_single(int n_frames, const float* in, int *out);
	void cv_bunch_single(int n_frames, const float* in, float *out);///////////////////////////////////////////by yongxu
	void returnWeights(float **weights, float **bias);    			/// copy weights and biases from gpu memory to cpu memory 

	int numlayers;
	int layersizes[MAXLAYER];
	int bunchsize;
	float lrate;
	float momentum;
	float weightcost;
	int dropoutflag;
		float visible_omit;
	float hid_omit;
private:
	void devnew_vf(const char* varname, int n, float **devptr);
	void devnew_vi(const char* varname, int n, int **devptr);/////////////////////////////by yongxu
	void devfree_vf(const char* varname,  float* devptr);
	void devfree_vi(const char* varname,  int* devptr);
	void todev_vf_vf(const char* varname, int n, const float* from, float* devto, cudaStream_t stream);
	void fromdev_vf_vf(const char* varname, int n, const float* devfrom, float* to, cudaStream_t stream);
	//void todev_vi_vi(const char* varname, int n, const int* from, int* devto, cudaStream_t stream);
	//void fromdev_vi_vi(const char* varname, int n, const int* devfrom, int* to, cudaStream_t stream);

	BP_WorkSpace *dev;  //viaribles for devices
	int GPU_total;							//devices used num, 表示所有GPU数目
	int GPU_selected;				//devices selected, 表示采用的GPU数目

	cublasHandle_t *handles;
	cudaStream_t *streams;
		curandGenerator_t *gen;
};
