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
#define MAXLAYERSIZE 20000
////
int   numlayers=8;
float beta=0.5;
int   ori_layersizes[MAXLAYER];
int   add_layersizes[MAXLAYER];
int   layersizes[MAXLAYER];
char  in_pfilename[MAXLINE];
char  out_pfilename[MAXLINE];
float *weights[MAXLAYER];
float *bias[MAXLAYER];
FILE  *fp_in;
FILE  *fp_out;

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
void Initial(int argc, char **argv)
{
	int i;
	char *p;
	char *argname;
	char *argvalue;
	char buff[MAXLINE];
	for(i =1; i < argc; i++)
	{
		p = strstr(argv[i],"=");
		if(p ==NULL)
		{
			printf("Error:Arg: %s  Format Error,please use '=' to connect name and value\n",argv[i]);
			exit(0);
		}
		argname = argv[i];
		argvalue = p+1;
		*p = '\0';
		if(0 == strcmp(argname, "numlayers"))
		{
			numlayers = atoi(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "beta"))
		{
			beta = atof(argvalue);
			continue;
		}
        if(0 == strcmp(argname, "in_pfilename"))
		{
			strcpy(in_pfilename, argvalue); 
			continue;
		}
		if(0 == strcmp(argname, "out_pfilename"))
		{
			strcpy(out_pfilename, argvalue); 
			continue;
		}
        if(0 == strcmp(argname, "ori_layersizes"))
		{
			char *pp = argvalue;
			int count =0;
			int j;
			p = strstr(argvalue, ",");
			while(p != NULL)
			{
				*p = '\0';
				ori_layersizes[count++] = atoi(pp);
				pp = p +1;
				p = strstr(pp, ",");
			}
			ori_layersizes[count++] = atoi(pp);
			continue;
		}
		if(0 == strcmp(argname, "add_layersizes"))
		{
			char *pp = argvalue;
			int count =0;
			int j;
			p = strstr(argvalue, ",");
			while(p != NULL)
			{
				*p = '\0';
				add_layersizes[count++] = atoi(pp);
				pp = p +1;
				p = strstr(pp, ",");
			}
			add_layersizes[count++] = atoi(pp);
			continue;
		}
	}
	for(i=0;i<numlayers;i++)
	{
		layersizes[i]=ori_layersizes[i]+add_layersizes[i];
	}
	////print info
	printf("parameters input:\n");
	printf("numlayers=%d\n",numlayers);
	printf("beta=%f\n",beta);
	printf("input pfile name=%s\n",in_pfilename);
	printf("output pfile name=%s\n",out_pfilename);
	printf("ori_layersizes:	 ");
	for(int j =0; j < numlayers; j++)
		printf("%d,", ori_layersizes[j]);
	printf("\n");
	printf("add_layersizes:  ");
	for(int j =0; j < numlayers; j++)
		printf("%d,", add_layersizes[j]);
	printf("\n");
    printf("layersizes:  ");
	for(int j =0; j < numlayers; j++)
		printf("%d,", layersizes[j]);
	printf("\n");
	////initial weights
	for(i =1; i< numlayers; i++)
	{
		int size	= layersizes[i] *layersizes[i-1];
		weights[i] = new float [size];
		bias[i] = new float [layersizes[i]];

		memset(weights[i],0,size *sizeof(float));
		memset(bias[i],0, layersizes[i] *sizeof(float));
	}
	if(NULL ==(fp_in = fopen(in_pfilename, "rb")))
	{
		printf("can not open output input pfile file: %s\n", in_pfilename);
		exit(0);
	}
	else
	{
		int m,n;
		int stat[10];
		char head[256];
        float *tmp_weights;
		float *tmp_bias;
	    tmp_weights = new float [MAXLAYERSIZE*MAXLAYERSIZE];
		tmp_bias = new float [MAXLAYERSIZE];
		memset(tmp_weights,0,(MAXLAYERSIZE*MAXLAYERSIZE)*sizeof(float));
		memset(tmp_bias,0,MAXLAYERSIZE *sizeof(float));

		for(i =1; i< numlayers; i++)
		{
			fread(stat,sizeof(int),5,fp_in);
			fread(head,sizeof(char),stat[4],fp_in);
			if(stat[1] != ori_layersizes[i] || stat[2] != ori_layersizes[i-1])
			{
				printf("original init weights node nums do not match\n");
				exit(0);
			}
            fread(tmp_weights,sizeof(float), ori_layersizes[i]* ori_layersizes[i-1],fp_in);

			fread(stat,sizeof(int),5,fp_in);
			fread(head,sizeof(char),stat[4],fp_in);
			if(stat[2] != ori_layersizes[i] || stat[1] != 1)
			{
			    printf("init bias node nums do not match\n");
				exit(0);
			}
			fread(tmp_bias,sizeof(float),ori_layersizes[i],fp_in); 

			///init weights and bias
			for(m=0;m<ori_layersizes[i-1];m++)
				for(n=0;n<ori_layersizes[i];n++)
			{
               weights[i][m*layersizes[i]+n]=tmp_weights[m*ori_layersizes[i]+n];
			}
			for(n=0; n<ori_layersizes[i];n++)
			{
				bias[i][n]=tmp_bias[n];
			}
		}
	    fclose(fp_in);
	    printf("Init weight file loaded over!.\n");
		delete [] tmp_weights;
		delete [] tmp_bias;
	}
}
void Writeweights()
{
  int stat[10];
  char head[256];
  int i;
  if(NULL ==(fp_out = fopen(out_pfilename, "wb")))
  {
	printf("can not open output pfile file: %s\n", out_pfilename);
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

		fwrite(stat,sizeof(int),5,fp_out);
		fwrite(head,sizeof(char),stat[4],fp_out);
		fwrite(weights[i],sizeof(float),stat[2] *stat[1],fp_out);
		
	    //biases
		sprintf(head,"bias%d",i+1);
		stat[0] = 10;
		stat[1] = 1;
		stat[2] =layersizes[i];
		stat[3] = 0;
		stat[4] = strlen(head)+1;

		fwrite(stat,sizeof(int),5,fp_out);
		fwrite(head,sizeof(char),stat[4],fp_out);
		fwrite(bias[i],sizeof(float),stat[2],fp_out);
	}
	printf("Saving over!\n");
	fclose(fp_out);
}

int main(int argc, char *argv[])
{
	int i,j;
	int m,n;
	float range;
	//char  outfilename[MAXLINE];
	if (argc==1)
	{
		cout<<"numlayers beta ori_layersizes add_layersizes in_pfile out_pfile"<<endl;
	}
	Initial(argc, argv);
	////gen random weights 
	for(i =1; i< numlayers ;i++)
	{
		int size = layersizes[i-1]*layersizes[i];
		//init weights with gaussian
		range = beta*sqrt(6.0f)/(sqrt(layersizes[i-1]+layersizes[i]));
	    printf("range=%f\n",range);  
		for(m=0;m<layersizes[i-1];m++)
			for(n=ori_layersizes[i];n<layersizes[i];n++)
		{
            weights[i][m*layersizes[i]+n]=range*uniformrand(-1.0f,1.0f);
		}
		for(m=ori_layersizes[i-1];m<layersizes[i-1];m++)
			for(n=0;n<ori_layersizes[i];n++)
		{
            weights[i][m*layersizes[i]+n]=range*uniformrand(-1.0f,1.0f);
		}
	}
	Writeweights();
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
  return 0;
}



	




