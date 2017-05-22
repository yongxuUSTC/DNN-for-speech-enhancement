clc
clear all;

scp_list='CleanTR08.SCP';
flsp=fopen(scp_list);
tline=fgetl(flsp);
system('mkdir clean_be');
line_num=0;
while(tline~=-1)
    line_num=line_num+1;
    
    old_tline=tline;
    out_tline=strrep(tline,'clean','clean_be');
    tline=old_tline;
    out_tline=strrep(out_tline,'.08','.08'); 
    
    le2be_for_all_files_func(tline, out_tline);
%     break;
tline=fgetl(flsp);
end