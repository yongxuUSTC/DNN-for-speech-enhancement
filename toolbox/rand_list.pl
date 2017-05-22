use strict;
use List::Util 'shuffle';

my ($ilist, $olist) = @ARGV;
my @list;
my @ind;
my @shuffled;
my $num;
my $i;

open(FILE_IN, "$ilist");
open(FILE_OUT,">$olist");
@list = <FILE_IN>;
$num = @list;
@ind = (0..$num-1);
@shuffled = shuffle(@ind);
foreach $i(@shuffled)
{
	print FILE_OUT $list[$i]; 
}

close(FILE_IN);
close(FILE_OUT);

