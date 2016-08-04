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
1. Please ref: DNN based speech enhancement tool is open now and can be downloaded at https://drive.google.com/file/d/0B5r5bvRpQ5DRXzJXd05BNl95alE/view

What else can this code use for?
1. It is designed for any regression tasks, like speech enhancement, ideal binary/ratio mask (IBM/IRM) estimation, audio/music tagging, acoustic event detection, etc.

Please cite the following papers if you use this code:
[1]A Regression Approach to Speech Enhancement Based on Deep Neural Networks.YongXu,JunDu,Li-Rong Dai and Chin-Hui Lee, IEEE/ACM Transactions on Audio,Speech, and Language Processing,P.7-19,Vol.23,No.1, 2015
[2]An Experimental Study on Speech Enhancement Based on Deep Neural Networks.YongXu, JunDu, Li-Rong Dai and Chin-Hui Lee,IEEE signal processing letters, p. 65-68,vol.21,no. 1,January 2014

Some DNN based speech enhancemen demos:
1. http://home.ustc.edu.cn/~xuyong62/demo/SE_DNN_taslp.html
2. http://home.ustc.edu.cn/~xuyong62/demo/IS15.html
