GPU code for Deep neural network (DNN) based speech enhancement

How to use?

1. make

2. use *.pl to call BPtrain

How to prepare the input and output files ?

1. use quicknet toolset to prepare Pfile as the input and the output files, Pfile is the big file of all training features.

What are the functions in this code ?

1. ReLU or Sigmoid

2. Noise aware training

3. Dropout

How to do decoding or speech enhancement in the test phase ?

1. Please ref: DNN based speech enhancement tool is open now and can be downloaded at https://drive.google.com/file/d/0B5r5bvRpQ5DRR1lIV1hpZ0RLQ0E/view?usp=sharing

or (@ Baidu Yun)
http://pan.baidu.com/s/1eRJGrx4 

What kinds of noisy speech can the DNN-enh tool enhance ?

1.  It can enhance any kinds of noisy speech, even the real-world noisy speech. one real-world noisy speech enh demo for the movie <Forrest Gump>: http://staff.ustc.edu.cn/~jundu/The%20team/yongxu/demo/IS15.html

2. The model is trained only on TIMIT data, so it can get the best performance on the TIMIT test set. 

3. The model can get the best performance on English dataset because TIMIT is US-English. But this tool still can be used to enhance the noisy speech in other languages, like Chinese.

4. You can use multi-language data to retrain this model to get a general DNN-enh tool.


What else can this code use for?

1. It is designed for any regression tasks, like speech enhancement, ideal binary/ratio mask (IBM/IRM) estimation, audio/music tagging, acoustic event detection, etc.

Please cite the following papers if you use this code:

[1]A Regression Approach to Speech Enhancement Based on Deep Neural Networks.YongXu,JunDu,Li-Rong Dai and Chin-Hui Lee, IEEE/ACM Transactions on Audio,Speech, and Language Processing,P.7-19,Vol.23,No.1, 2015

[2]An Experimental Study on Speech Enhancement Based on Deep Neural Networks.YongXu, JunDu, Li-Rong Dai and Chin-Hui Lee,IEEE signal processing letters, p. 65-68,vol.21,no. 1,January 2014

[3] Multi-Objective Learning and Mask-Based Post-Processing for Deep Neural Network Based Speech Enhancement, Yong Xu, Jun Du, Zhen Huang, Li-Rong Dai, Chin-Hui Lee, Interspeech2015

Some DNN based speech enhancemen demos:

http://staff.ustc.edu.cn/~jundu/The%20team/yongxu/demo/SE_DNN_taslp.html
http://staff.ustc.edu.cn/~jundu/The%20team/yongxu/demo/IS15.html
