#include <sys/time.h>
#include <math.h>

#define MAXLAYER 10
#define MAXLINE 1024
#define MAXCHUNK 102400

struct WorkPara
{
	char fea_FN[MAXLINE];
	char fea_normFN[MAXLINE];	
	int fea_dim;
	int fea_context;

	char targ_FN[MAXLINE];
	int targ_offset;
	int dropoutflag;
	int traincache;        ////frames to memory one time
	int bunchsize;
	int layersizes[MAXLAYER];
	float momentum;
	float weightcost;
	float lrate;
	float visible_omit;
	float hid_omit;
	
	char init_weightFN[MAXLINE];
	char out_weightFN[MAXLINE];
	char log_FN[MAXLINE];
	
	char train_sent_range[MAXLINE];
	char cv_sent_range[MAXLINE];
	
	int gpu_used;
	int init_randem_seed;
	float init_randem_weight_min;
	float init_randem_weight_max;
	float init_randem_bias_max;
	float init_randem_bias_min;
	
	float *indata;
	//int *targ;
	float *targ;///////////////////////////////////////////////////////////////////////////by yongxu
	float *weights[MAXLAYER -1];
	float *bias[MAXLAYER -1];
};

class Interface
{
public:
		Interface();
		~Interface();
public:
		void Initial(int argc, char **argv);
		void Writeweights();
		int  Readdata();
		void get_pfile_info();
		void get_chunk_info(char *range);
		void get_chunk_info_cv(char *range);
		int Readchunk(int index);
		int Readchunk_cv(int index);
		void GetRandIndex(int *vec, int len);
public:
		struct WorkPara *para;
		
		unsigned int total_frames;
		unsigned int total_sents;
		unsigned int total_chunks;
		unsigned int total_samples;
		unsigned int cv_total_chunks;
		unsigned int cv_total_samples;
		
		int *framesBeforeSent;
		int *chunk_frame_st;
		int *cv_chunk_frame_st;
		
		FILE *fp_log;
		int numlayers;
		int realbunchsize;
private:
		void get_uint(const char* hdr, const char* argname, unsigned int* val);
		void read_tail(FILE *fp, long int file_offset, unsigned int sentnum, int *out);
		
		void GetRandWeight(float *vec, float min, float max, int len);
		
		FILE *fp_data;
		FILE *fp_targ;
		FILE *fp_init_weight;
		FILE *fp_norm;
		FILE *fp_out;
		
		int data_rand_index[MAXLINE];
		
		float *mean;
		float *dVar;
		
		int sent_st, sent_en;
		int cv_sent_st, cv_sent_en;
		int cur_chunk_index;
		int frames_read;
};
