#!/bin/bash
day=''
usage="$0 --day DAY"
# Parse command line options
while (( "$#" )); do
    case "$1" in
        --day) day=$2; shift 2;;
        *)
            echo "Usage: $usage"; exit 1;;
    esac
done
# Load software
module load pytorch/v1.6.0
# Run algorithm
$HOME/mldas/scripts/probmap/wrapper.sh /global/cscratch1/sd/vdumont/1min_ch4650_4850/westSac_1801$day
