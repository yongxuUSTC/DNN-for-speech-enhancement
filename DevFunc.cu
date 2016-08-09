#include "DevFunc.h"
#include <stdlib.h>

__global__ void kernBinary(int n, float* in_vec, float* rand_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
    {
			if(in_vec[i] > rand_vec[i])
			{
				in_vec[i] = 1.0f;
			}
			else
			{
				in_vec[i] = 0.0f;
			}
		}
}

__global__ void kernWeightMultiP( int n, float p, float* in_vec )
{
//	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
//	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
//	if(i < prev_n&& j < cur_n)
//	{
//	   in_vec[i+cur_n*j] = in_vec[i+cur_n*j]*p;
//	}
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if( i < n )
	{
		in_vec[i]=in_vec[i]*p;
	}
}
__global__ void kernDropout(int n, float p ,float* in, float* rand_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
	{
	   if(rand_vec[i]<p)
	   {
		   in[i]=0;
	   }
	}

}

__global__ void kernSigmoid(int n, float* in_vec, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
			out_vec[i] = 1.0f/(1.0f + expf(- in_vec[i]));
}

__global__ void kernDsigmoid(int n, float* in_vec, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
    {
			const float y = in_vec[i];
			out_vec[i] = (1.0f - y) * y;
    }
}

__global__ void  kernSoftmax(int rows, int cols, float* in_vec, float* out_vec)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row < rows)
    {
			int i;
			const int index = row * cols;
			const float* invec = &in_vec[index];
		  float* outvec = &out_vec[index];
			const float* inptr;
			float* outptr;
		
			// First find the max of each vector
			float max;
			
			inptr = invec;
			max = *inptr++;
			for (i=cols-1; i!=0; i--)
			{
			    float val;
		
			    val = *inptr++;
			    if (val>max)
				max = val;
			}
			// Now put exp(in-max) in out
			inptr = invec;
			outptr = outvec;
			float sumexp = 0;
			for (i=cols; i!=0; i--)
			{
			    float f, e;
			    
			    f = *inptr++;
			    e = expf(f - max);
			    *outptr++ = e;
			    sumexp += e;
			}
			// Now scale the output
			float scale = 1.0f/sumexp;
			outptr = outvec;
			for (i=cols; i!=0; i--)
			{
			    *outptr = (*outptr) * scale;
			    outptr++;
			}
    }
}

__global__ void  kernLinearOutCopy(int rows, int cols, float* in_vec, float* out_vec)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row < rows)
    {
			//int i; //xuyong
			//const int index = row * cols;
			//const float* invec = &in_vec[index];
		  //float* outvec = &in_vec[index];
		  ////////////////////////////////////////////////////
		 int j;
	 	 for(j =0; j< cols;j++)
		 	out_vec[cols *row +j] = in_vec[cols *row +j];
		 	
    }
}

__global__ void kernMultiCopy(int mat_height, int vec_len,
		   float* vec, float* mat)
{
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (col < vec_len)
    { 
			int j;
			float val = vec[col];
			float* top = &mat[col];
			for (j=mat_height; j!=0; j--)
			{
			    *top = val;
			    top += vec_len;
			}
    }
}

__global__ void kernSumcol(int rows, int cols, float* in, float* res)
{
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (col < cols)
    {
			int j;
			const float* fromp = &in[col];
			float* top = &res[col];
			
			(*top) = (*fromp);
			fromp +=cols;
			for (j=rows-1; j!=0; j--)
			{
			    (*top) += (*fromp);
			    fromp+=cols;
			}
    }
}

__global__ void kernAccSumcol(int rows, int cols, float* in, float* res, float alpha, float beta)
{
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (col < cols)
    {
			int j;
			const float* fromp = &in[col];
			float* top = &res[col];
			
			(*top) = (*top) *alpha + beta *(*fromp);
			fromp +=cols;
			for (j=rows-1; j!=0; j--)
			{
			    (*top) += beta *(*fromp);
			    fromp+=cols;
			}
    }
}

__global__ void kernAccSumrow(int rows, int cols, float* in, float* res, float alpha, float beta)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (row < rows)
    {
			int j;
			const float* fromp = &in[row];
			float* top = &res[row];
			
			(*top) = (*top) *alpha + beta *(*fromp);
			fromp +=rows;
			for (j= cols -1; j!=0; j--)
			{
			    (*top) += beta *(*fromp);
			    fromp += rows;
			}
    }
}

__global__ void kernVecMul(int n, float* in_vec1, float* in_vec2, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
			out_vec[i] = in_vec1[i] * in_vec2[i];
}

//__global__ void kernSubIndex( int rows , int cols, const float *in_vec1, const int *in_index, float *res_vec)
__global__ void kernSubClean( int rows , int cols, const float *in_vec1, const float *in_clean, float *res_vec)
{
	 int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	 if(i < rows)
	 {
	 	 int j;
	 	 for(j =0; j< cols;j++)
		 	{ 	//res_vec[cols *i +j] = in_vec1[cols *i +j];
		 //int ind = in_index[i];
		 //res_vec[cols *i + ind] = in_vec1[cols *i +ind] - 1.0f;
		 res_vec[cols *i + j] = (2.0f/rows)*(in_vec1[cols *i +j]-in_clean[cols *i +j]);
		 //res_vec[cols *i + j] = 2.0f*(in_vec1[cols *i +j]-in_clean[cols *i +j]);
		 //printf("in kernSubClean, res_vec=%f ",res_vec[cols *i + j]);
		 }
	 }
}

__global__ void kernAccSum(int n, float* in, float* res, float beta)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i <n)
	{
		res[i] = in[i] + beta *res[i];
	}
}

//__global__ void kernGetMaxIndex(int rows, int cols, float* invec, int* outvec)
//{
//	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
//	if(i < cols)
//	{
//		float *p = invec + rows * i;
//		int maxinx = 0;
//		float max = *p;
//		for(int j=1;j< rows;j++)
//		{
//			if(p[j] > max)
//			{
//				max = p[j];
//				maxinx = j;
//			}
//		}
//		outvec[i] = maxinx;
//	}
//}

__global__ void kernDivide(int n, float* in_vec, float* out_vec,float beta)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
			out_vec[i] = in_vec[i]/beta;
}

//__global__ void kernUpdatedelta(int size, float* delta, float* weights, float* gradient, int n, float momentum, float lr, float weightcost)
//{
//    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
//    if (i < size)
//			delta[i] = momentum * delta[i] - lr * (gradient[i] / n + weightcost * weights[i]);
//}

__global__ void kernUpdatedelta(int size, float* delta, float* weights, float* gradient, int n, float momentum, float lr, float weightcost)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size)
			delta[i] = momentum * delta[i] - (1-momentum)*lr*(gradient[i] / n + weightcost * weights[i]);//3.16 dropoutÊ±Òª³Ë1-momentum
}
