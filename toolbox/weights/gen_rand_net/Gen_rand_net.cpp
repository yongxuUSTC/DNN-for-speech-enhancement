#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <math.h>
#include <time.h>

using namespace std;

#define PI 3.14159
#define MAXLAYER 10
#define MAXLINE 1024

float uniformrand(float DOWN,float UP)
{

	int min = (int)(DOWN*1000000);
	int max = (int)(UP*1000000);
	int randint = rand();
	int diff = max-min;
	int result = randint%diff+min;
	return result/1000000.0;
}

float Normal(float x,float miu,float sigma) //正态分布概率密度函数
{
	return (float)(1.0/(sqrt(2.0*PI)*sigma))*exp(-1.0*(x-miu)*(x-miu)/(2.0*sigma*sigma));
}

float guassrand()
{
	//srand(unsigned(time(NULL)));
	float x;
	float dScope;
	float y;
	do
	{
		x = uniformrand(-1.0f,1.0f); 
		y = Normal(x, 0, 1);
		dScope = uniformrand(0, Normal(0,0,1));
	}while( dScope > y);
	return x;
}

int main(int argc, char *argv[])
{
	int   i,j;
	int   numlayers;
	int   layersizes[MAXLAYER];
	int   flag;
	float beta;
	float range;
	char  out_dir[MAXLINE];
	char  outfilename[MAXLINE];
	char  out_pfilename[MAXLINE];
	char  out_info_file[MAXLINE];
	float *weights[MAXLAYER -1];
	float *bias[MAXLAYER -1];
	FILE  *fp;
	if (argc==1)
	{
		cout<<"numlayers layersizes[0] layersizes[1] ... layersizes[numlayers-1] out_dir out_pfile flag beta"<<endl;
		cout<<"flag=0, weights initialzed to uniform distribution U(-1/sqrt(n), 1/sqrt(n))"<<endl;
		cout<<"flag=1, weights initialzed to uniform distribution U(-sqrt(6)/(sqrt(n_i+n_j), sqrt(6)/(sqrt(n_i+n_j))"<<endl;
		return 0;
	}
	numlayers=atoi(argv[1]);
	printf("numlayers=%d\n",numlayers);
	printf("layersizes:");
	for (i=0; i<numlayers; i++)
	{
		layersizes[i]=atoi(argv[i+2]);
        printf("%d, ",layersizes[i]);
	}
	printf("\n");
	strcpy(out_dir,argv[numlayers+2]);
	strcpy(out_pfilename,argv[numlayers+3]);
    flag= atoi (argv[numlayers+4]);
	beta= atof (argv[numlayers+5]);
	printf("flag=%d\n",flag);
	////gen random weights and biases
	for(i =1; i< numlayers ;i++)
	{
		int size = layersizes[i-1]*layersizes[i];
		//init weights with gaussian
		weights[i-1] = new float [size];
		if(flag)
		   range = beta*sqrt(6.0f)/(sqrt(layersizes[i-1]+layersizes[i]));
		else
           range = beta*1.0f/sqrt(layersizes[i-1]);
	    printf("range=%f\n",range);   
		for (j=0; j<size; j++)
		{
			weights[i-1][j] = range*uniformrand(-1.0f,1.0f);
		}
	
		//init biases with 0
		bias[i-1] =new float[layersizes[i]];
		memset(bias[i-1],0,layersizes[i] *sizeof(float));
		//save
//		sprintf(outfilename,"%s/rbm%d.rand.wts",out_dir,i);
//		if(NULL ==(fp = fopen(outfilename, "wb")))
//		{
//			printf("cannot open %s for write\n",outfilename);
//			exit(0);	
//		}
//		fwrite(weights[i-1],sizeof(float),size,fp);
//		fwrite(bias[i-1],  sizeof(float),layersizes[i],fp);
//		printf("rbm%d.rand.wts: %d-%d\n",i,layersizes[i-1],layersizes[i]);
	}
////	///////test
//	for (i =1; i< numlayers ;i++)
//	{
//		sprintf(outfilename,"%s/rbm%d.rand.txt",out_dir,i);
//		if(NULL ==(fp = fopen(outfilename, "w")))
//		{
//			printf("cannot open %s for write\n",outfilename);
//			exit(0);	
//		}
//		fprintf(fp,"rbm%d.rand.wts weights: %d-%d\n",i,layersizes[i-1],layersizes[i]);
//		for (j=0; j<layersizes[i-1]*layersizes[i];j++)
//		{
//            fprintf(fp,"%10.8f  ",weights[i-1][j]);
//			if ((j+1)%layersizes[i]==0)
//				fprintf(fp,"\n");
//		}
//		fprintf(fp,"rbm%d.rand.wts biases: %d\n",i,layersizes[i]);
//		for (j=0; j<layersizes[i];j++)
//		{
//			fprintf(fp,"%10.8f  ",bias[i-1][j]);
//		}
//	   fprintf(fp,"\n");
//	}
//  fclose(fp);

  int stat[10];
  char head[256];
  if(NULL ==(fp= fopen(out_pfilename, "w")))
  {
		printf("cannot open %s for write\n",out_pfilename);
		exit(0);	
  }
  for(i =1; i< numlayers; i++)
  {
	    ///weights
		sprintf(head,"weights%d%d",i,i+1);
		stat[0] = 10;
		stat[1] =layersizes[i];
		stat[2] =layersizes[i -1];
		stat[3] = 0;
		stat[4] = strlen(head)+1;

		fwrite(stat,sizeof(int),5,fp);
		fwrite(head,sizeof(char),stat[4],fp);
		fwrite(weights[i-1],sizeof(float),stat[2] *stat[1],fp);
		
	    //biases
		sprintf(head,"bias%d",i+1);
		stat[0] = 10;
		stat[1] = 1;
		stat[2] =layersizes[i];
		stat[3] = 0;
		stat[4] = strlen(head)+1;

		fwrite(stat,sizeof(int),5,fp);
		fwrite(head,sizeof(char),stat[4],fp);
		fwrite(bias[i-1],sizeof(float),stat[2],fp);
	}
	printf("Saving over!\n");
	fclose(fp);
  return 0;
}



	




