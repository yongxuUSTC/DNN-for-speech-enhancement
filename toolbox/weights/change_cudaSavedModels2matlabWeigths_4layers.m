clc
clear all;

for i=25

mat_name=strcat('mlp.',num2str(i),'.wts.mat');
load(mat_name);
% break;

w1=[weights12 bias2'];
w1=w1';
w2=[weights23 bias3'];
w2=w2';
w3=[weights34 bias4'];
w3=w3';
% w4=[weights45 bias5'];
% w4=w4';
% w5=[weights56 bias6'];
% w5=w5';
% w6=[weights67 bias7'];
% w6=w6';


wts_name=sprintf('se_weights%d',i);
% save (wts_name,'w1','w2');
save (wts_name,'w1','w2','w3');
% save (wts_name,'w1','w2','w3','w4');
% save (wts_name,'w1','w2','w3','w4','w5');
% save (wts_name,'w1','w2','w3','w4','w5','w6');

 clearvars -EXCEPT i;
end;

 clear all;