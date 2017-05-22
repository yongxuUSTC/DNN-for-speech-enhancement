
% function []=le2be_for_all_files_func(infile, outfile)
infile='clean_FBI_22123A.08';
outfile='clean_FBI_22123A.08_be';

fn = fopen(infile, 'r','ieee-le');        
Y = fread(fn, inf, 'short');  
fclose(fn); %%%关闭当前文件句柄，否则最后会提示打开了太多文件

fid = fopen(outfile,'wb','ieee-be');
fwrite(fid,Y,'short');
fclose(fid);
% end