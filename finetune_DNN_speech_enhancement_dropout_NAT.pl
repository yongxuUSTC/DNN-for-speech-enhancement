use strict;

my $i;
my $j;
my $line;
my $curacc;
my $preacc;
#my $threshold=0.1;


my $numlayers=5;

	my $lrate=1;
	my $layersizes = "1548"; # 129*11+129
	for(my $i=0;$i<$numlayers -2;$i++)
	{
		$layersizes	  .= ",2048";
	}	
	$layersizes	  .= ",129";########################
	
	my $node=2048;
	
#	my $hidname = "";
#	for(my $i=0;$i<$numlayers -2;$i++)
#	{
#		$hidname	  .= "_h500";
#	}	

	my $exe 						= "./code_BP_GPU_DNN_Dropout_NAT_speech_enhancement_GPU1/BPtrain";
	my $gpu_used				= 1;
#	my $numlayers				= 4;
#	my $layersizes			= "429,1024,1024,183";

	my $bunchsize				= 128;#128
	my $momentum				= 0.5;
	my $weightcost			= 0;
	my $fea_dim					= 129;#123
	my $fea_context			= 11;
	my $traincache			= 102400;  ############ how many samples per chunk #102400
	my $init_randem_seed= 27863875;   ############ every epoch must change
	my $targ_offset			= 5;
	
#	my $CF_DIR					= "config";
#	my $norm_file				= "$CF_DIR/fea_tr.norm_data_timit_SNR_20_15_10_5_0_-5";
#	my $fea_file				= "$CF_DIR/timit_Multi_NT_SNR_100h_all_trainset_25cases_random_ts2000_noisy.pfile";
#	my $targ_file				= "$CF_DIR/timit_Multi_NT_SNR_100h_all_trainset_25cases_random_ts2000_clean.pfile";########################
	my $CF_DIR					= "/home/yongxu/step1_prepare_data/data_timit_104NT_7SNRs_100h_phase_from18/pretrain_pfile";
	my $norm_file				= "$CF_DIR/104NT_7SNRs_2500h_EachCase4H_trainset_random_ts2500.fea_norm";
	my $fea_file				= "$CF_DIR/104NT_7SNRs_100h_EachCase4H_trainset_random_ts2500_noisy_linux.pfile";
	my $targ_file				= "/disk1/yongxu_d1/config/get_100h_104NT_7SNRs_random_ts2500/104NT_7SNRs_100h_EachCase4H_trainset_estIBM_refCLEAN_LC5dB_random_ts2500_noisy_linux.pfile";########################
		
#	my $train_sent_range		= "0-115499";
#	#my $train_sent_range		= "8-9";
#	my $cv_sent_range				= "115500-117499";
#	#my $cv_sent_range				= "1-1";
#	my $train_sent_range		= "0-721874"; #625h
#	#my $train_sent_range		= "8-9";
#	my $cv_sent_range				= "2887500-2889999";
#	#my $cv_sent_range				= "1-1";
	#my $train_sent_range		= "0-2887499";#2500h
#	my $train_sent_range		= "0-1443749"; #1250h
#	#my $train_sent_range		= "8-9";
#	my $cv_sent_range				= "2887500-2889999";
#	#my $cv_sent_range				= "1-1";
	my $train_sent_range		= "0-115499"; #100h
	#my $train_sent_range		= "8-9";
	my $cv_sent_range				= "115500-117999";
	#my $cv_sent_range				= "1-1";
	
	my $MLP_DIR					= "models/104NT_7SNRs_100h_EachCase4H_trainset_random_ts2500_batch$bunchsize\_momentum$momentum\_frContext$fea_context\_lrate$lrate\_node$node\_numlayer$numlayers\-randomPretr-v0.1-h0.2-dropout-NAT-estIBM_refCLEAN_LC5dB-GPU1";###########################################################################
	
	system("mkdir $MLP_DIR");
	my $outwts_file			= "$MLP_DIR/mlp.1.wts";
	my $log_file				= "$MLP_DIR/mlp.1.log";
	my $initwts_file		= "pretraining_weights/random_1548_2048_2048_2048_129.wts";#########################
	###my $initwts_file		= "/home/jiapan/new_BP_Code/BPtrain_v1_mlp/mlp.6.wts.right";
#	
	#printf("2");
	print "iter 1 lrate is $lrate\n"; 
	system("$exe" .
		" gpu_used=$gpu_used".
		" numlayers=$numlayers".
		" layersizes=$layersizes".
		" bunchsize=$bunchsize".
		" momentum=$momentum".
		" weightcost=$weightcost".
		" lrate=$lrate".
		" fea_dim=$fea_dim".
		" fea_context=$fea_context".
		" traincache=$traincache".
		" init_randem_seed=$init_randem_seed".
		" targ_offset=$targ_offset".
		" initwts_file=$initwts_file".
		" norm_file=$norm_file".
		" fea_file=$fea_file".
		" targ_file=$targ_file".
		" outwts_file=$outwts_file".
		" log_file=$log_file".
		" train_sent_range=$train_sent_range".
		" cv_sent_range=$cv_sent_range".
		" dropoutflag=1".
		" visible_omit=0.1".
		" hid_omit=0.2"
		);
		
#	die;
#	
#		my $success=open LOG, "$log_file";
#		if(!$success)
#		{
#			printf "open log fail\n";
#		}
#		while(<LOG>)
#	  {
#	  	chomp;
#	  	if(/CV over.*/)
#	  	{
#	  	  s/CV over\. right num: \d+, ACC: //; 
#	  	  s/%//; 
#	  	  $curacc=$_;
#	  	}	  	
#	  }
#	  close LOG;
#	  
  $preacc=$curacc;
	my $destep=0;
	########################################
#	$init_randem_seed=27865600;
#	$momentum=0.7;
	########################################
	for($i= 2;$i <= 10;$i++){

		$j = $i -1;
		$initwts_file		= "$MLP_DIR/mlp.$j.wts";
		$outwts_file		= "$MLP_DIR/mlp.$i.wts";
		$log_file				= "$MLP_DIR/mlp.$i.log";
		$init_randem_seed  += 345;
    $momentum=$momentum+0.04;
    print "iter $i lrate is $lrate\n"; 
		system("$exe" .
		" gpu_used=$gpu_used".
		" numlayers=$numlayers".
		" layersizes=$layersizes".
		" bunchsize=$bunchsize".
		" momentum=$momentum".
		" weightcost=$weightcost".
		" lrate=$lrate".
		" fea_dim=$fea_dim".
		" fea_context=$fea_context".
		" traincache=$traincache".
		" init_randem_seed=$init_randem_seed".
		" targ_offset=$targ_offset".
		" initwts_file=$initwts_file".
		" norm_file=$norm_file".
		" fea_file=$fea_file".
		" targ_file=$targ_file".
		" outwts_file=$outwts_file".
		" log_file=$log_file".
		" train_sent_range=$train_sent_range".
		" cv_sent_range=$cv_sent_range".
	  " dropoutflag=1".
		" visible_omit=0.1".
		" hid_omit=0.2"
		);
	}
		
#		my $success=open LOG, "$log_file";
#		if(!$success)
#		{
#			printf "open log fail\n";
#		}
#		while(<LOG>)
#	  {
#	  	chomp;
#	  	if(/CV over.*/)
#	  	{
#	  	  s/CV over\. right num: \d+, ACC: //; 
#	  	  s/%//; 
#	  	  $curacc=$_;
#	  	}	  	
#	  }
#	  close LOG;
#
#	  if($curacc<$preacc+$threshold)	
#	  {
#	  	print "iter $i ACC $curacc < iter $j ACC $preacc+threshold($threshold)\n";
#	  	$destep++;
#	  	print "destep is $destep\n";
#	  	if($destep>=3)
#	  	{
#	  		
#	  		unlink($outwts_file) or warn "can not delete weights file";
#	  		unlink($log_file) or warn "can not delete log file";
#	  		$i+100;
#	  		#print "finetune end\n";
#	  		last;
#	  	}
#	  	else
#	  	{
#	  	$i--;	  	
#	  	$lrate *=0.5;
# 	    }
#	  }
#	  else
#	  {
#	  	$destep=0;
#	  	$preacc=$curacc;
#	  	print "1\n\n\n\n\n\n\n\n";
#	  }
#
#	}
#	
##
	for($i= 11;$i <= 100;$i++){
		$j = $i -1;
		$initwts_file		= "$MLP_DIR/mlp.$j.wts";
		$outwts_file		= "$MLP_DIR/mlp.$i.wts";
		$log_file				= "$MLP_DIR/mlp.$i.log";
		#$lrate *= 0.9;
		#$lrate = 0.01;
		$momentum=0.9;
		$init_randem_seed  += 345;
		
		system("$exe" .
		" gpu_used=$gpu_used".
		" numlayers=$numlayers".
		" layersizes=$layersizes".
		" bunchsize=$bunchsize".
		" momentum=$momentum".
		" weightcost=$weightcost".
		" lrate=$lrate".
		" fea_dim=$fea_dim".
		" fea_context=$fea_context".
		" traincache=$traincache".
		" init_randem_seed=$init_randem_seed".
		" targ_offset=$targ_offset".
		" initwts_file=$initwts_file".
		" norm_file=$norm_file".
		" fea_file=$fea_file".
		" targ_file=$targ_file".
		" outwts_file=$outwts_file".
		" log_file=$log_file".
		" train_sent_range=$train_sent_range".
		" cv_sent_range=$cv_sent_range".
		" dropoutflag=1".
		" visible_omit=0.1".
		" hid_omit=0.2"
		);
	}
