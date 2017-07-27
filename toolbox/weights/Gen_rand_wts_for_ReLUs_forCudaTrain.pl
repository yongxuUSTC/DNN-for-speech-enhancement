#!/usr/bin/perl
use strict;
use warnings;

my $i;
my $numlayers   = 5;
my $beta        = 0.5;
my $flag        = 1;
my $root_dir    = "/disk4/yongxu_d4/step2_BP_GPU_timit_104NT/gen_rand_net";
my $fname       = "Rand_2056_3hid2048_284.belta$beta";
my $out_wts_dir = "$root_dir/pretraining_weights/$fname";
my $out_pfilename = "$out_wts_dir/$fname.wts";
system("mkdir $out_wts_dir");
my $cmd = "$root_dir/Gen_rand_net $numlayers 2056 2048 2048 2048 284 $out_wts_dir $out_pfilename $flag $beta";
system($cmd);
 $cmd = "cp $out_wts_dir/$fname.wts pretraining_weights/.";
system($cmd);

