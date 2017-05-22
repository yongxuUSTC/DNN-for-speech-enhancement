use warnings;

open(IN1,"timit_aurora4_all_lsp_be.scp");
open(IN2,"timit_aurora4_all_lsp_be.len");
while(<IN1>)
{
	chomp;
$scp1=$_;
$len1=<IN2>;
$len_info{$scp1}=$len1;
}
close(IN1);
close(IN2);

open(IN3,"timit_aurora4_102NT_7SNRs_each190_80utts_noisy_lsp_be_random.scp");
open(OUT1,">timit_aurora4_102NT_7SNRs_each190_80utts_noisy_lsp_be_random.len");
while(<IN3>)
{
	###timit
	if(/TIMIT_16kHz_SEDNN_data_100h_100h.+_be\\(T.+_DR[\d]_.+_.+)\.lsp/)
	{
		print OUT1 "$len_info{$1}";
		next;
		}
	###aurora4
	if(/SE_100h\\NoisyFeature\\N[\d]+_SNR[\d\-]+\\(.+_16k\\.+)\.lsp_b.+/)
	{
	  print OUT1 "$len_info{$1}";
		next;
		}
	}
	close(IN3);
	close(OUT1);
	system("pause\n");