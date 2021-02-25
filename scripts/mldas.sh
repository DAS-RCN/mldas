#!/bin/bash

#=========================================================================================
# Path to software directory
#=========================================================================================

MLDAS=$(dirname -- $(dirname -- $(realpath "$0")))
SCRATCH=/global/scratch/vdumont/

#=========================================================================================
# Help message
#=========================================================================================

Help()
{
    echo
    echo "MLDAS wrapper for DAS data analysis."
    echo
    echo "Usage: $(basename "$0") -a action -p platform"
    echo "       [-f fmt] [-s soft] [-d data] [-w prob] [--mpi]"
    echo
    echo "-a,--action      Operation to be executed. Available options:" 
    echo "                   arrayudf ... Install ArrayUDF software"
    echo "                   convert .... Convert MAT files to HDF5"
    echo "                   dassa ...... Install DASSA software"
    echo "                   octave ..... Install Octave language"
    echo "                   probmap .... Calculate probability map"
    echo "                   pws ........ Perform phase-weighted stack"
    echo "                   rmsd ....... Calculate RMSD of each stack"
    echo "                   xcorr ...... Cross-correlate raw DAS data"
    echo
    echo "-p,--platform    Platform to be used. Available options:" 
    echo "                   colab ...... Google Colaboratory notebook"
    echo "                   lawrencium . LBNL Lawrencium cluster"
    echo "                   cori ....... NERSC Cori supercomputer"
    echo
    echo "-s,--software    Software to be used for phase-weighted stack:" 
    echo "                   matlab ..... MATLAB software"
    echo "                   dassa ...... Bin Dong's DASSA software"
    echo
    echo "-d,--data        Path to input target data. If the path contains"
    echo "                 an asterisk, please use quotes around the string."
    echo
    echo "-o,--out         Path to output repository where data are saved."
    echo
    echo "-w,--weight      Path to probability map."
    echo
    echo "--mpi            Number of CPUs for parallel execution."
    echo
}

#=========================================================================================
# Parse command line options
#=========================================================================================

input_args=$@
out=$SCRATCH
extra="none"
freqs="0.002:0.006:14.5:15"
while (( "$#" )); do
    case "$1" in
        -a | --action)   action=$2   ;;
        -c | --count)    count=$2    ;;
        -d | --data)     data=$2     ;;
        -e | --extra)    extra=$2    ;;
        -f | --freqs)    freqs=$2    ;;
	-h | --help)     Help; exit  ;;
	-m | --mpi)      mpi=$2      ;;
        -o | --out)      out=$2      ;;
        --order)         order=$2    ;;
        -p | --platform) platform=$2 ;;
	-s | --software) software=$2 ;;
	-w | --weight)   weight=$2   ;;
        *) echo "Invalid arguments. Abort."; exit 1
    esac
    shift
    shift
done
freqs=(${freqs//:/ })

#=========================================================================================
# Check that action, platform and data are specified
#=========================================================================================

valid_platforms=("colab" "lawrencium" "cori")
if [[ -z "$action" ]]
then
    echo "Action not specified. Abort."; exit
elif [[ -z "$data" ]]
then
    echo "Path to data not specified. Abort."; exit
elif ! [[ ${valid_platforms[*]} =~ "$platform" ]]
then
    echo "Platform not specified or not valid. Abort."; exit
fi

#=========================================================================================
# Purge module if access to cluster
#=========================================================================================

if [[ "$platform" != "colab" ]]
then
    module purge
fi

#=========================================================================================
# If requested (using -m/--mpi option), send SLURM job to supercomputer
#=========================================================================================

if [[ ( ! -z $mpi ) && ( $mpi != 0 ) ]]
then
    mkdir -p logs
    sed "s,ntasks,${mpi},g" $MLDAS/scripts/mpi_lrc.sh > mldas_tmp.sh
    if [[ $action == pws ]]
    then
	if [[ $software == dassa ]]
	then
	    filelist=(${data}/*.h5)
	elif [[ $software == matlab ]]
	then
	    filelist=(${data}/*.mat)
	fi
	num="${#filelist[@]}"
	for ((x=1; x<=$num; x++));
	do
	    sbatch mldas_tmp.sh "$input_args" -c $x
	done
    elif [[ $action == xcorr ]]
    then
	for input in ${data}
	do
	    new_input="${input_args/$data/$input}"
	    sbatch mldas_tmp.sh $new_input
 	done
    elif [[ $action == weight ]]
    then
	for input in ${data}/*.mat
	do
	    new_input="${input_args/$data/$input}"
            probmap=$(basename -- $input)
	    probmap=${probmap/.mat/.txt}
	    new_input="${new_input/$weight/$weight/$probmap}"
	    sbatch mldas_tmp.sh $new_input
	done
    else
	sbatch mldas_tmp.sh "$input_args"
    fi
    rm mldas_tmp.sh
    exit
fi

#========================================================================================
# Install ArrayUDF to produce cross-correlated data readable with MATLAB scripts
#========================================================================================

if [[ "$action" == "arrayudf" ]]
then
    cd $HOME
    if [ -d "ArrayUDF" ]; then rm -rf ArrayUDF; fi
    if [ -d "hdf5-1.10.5" ]; then rm -rf hdf5-1.10.5; fi
    git clone https://bitbucket.org/dbin_sdm/arrayudf-test/src/master/ ArrayUDF
    if [[ "$platform" == "cori" ]]
    then
	echo "ArrayUDF installation not implemented on Cori yet. Abort."
	exit
    elif [[ "$platform" == "colab" ]]
    then
	apt-get install libtool mpich wget automake libfftw3-dev
	wget https://s3.amazonaws.com/hdf-wordpress-1/wp-content/uploads/manual/HDF5/HDF5_1_10_5/source/hdf5-1.10.5.tar
	tar -xvf hdf5-1.10.5.tar && rm -rf hdf5-1.10.5.tar
	cd hdf5-1.10.5 && ./configure --enable-parallel --prefix=/usr CC=mpicc && make install
	cd $HOME/ArrayUDF && autoreconf -i && ./configure --with-hdf5=/usr CC=mpicc CXX=mpicxx --prefix=$HOME/ArrayUDF
	make clean && make install    
    elif [[ "$platform" == "lawrencium" ]]
    then
	module load gcc/7.4.0 hdf5/1.10.5-gcc-p fftw boost
	H5_PATH="/global/software/sl-7.x86_64/modules/gcc/7.4.0/hdf5/1.10.5-gcc-p"
	cd $HOME/ArrayUDF && autoreconf -i && ./configure --with-hdf5=$H5_PATH CC=mpicc CXX=mpicxx --prefix=$HOME/ArrayUDF
	make clean && make install
    fi
    cd $HOME/ArrayUDF/examples/das
    sed -i 's,../../build,../..,g' Makefile
    sed -i 's,-mt,,g' Makefile
    make das-fft-full
fi

#========================================================================================
# Install DASSA software
#========================================================================================
    
if [[ "$action" == "dassa" ]]
then
    if [[ "$platform" == "lawrencium" ]]
    then
	module load gcc/7.4.0 hdf5/1.10.5-gcc-p fftw boost
	cd $HOME
	git clone https://bitbucket.org/dbin_sdm/dassa/src/master/ dassa
	cd dassa && make
    fi
fi

#=========================================================================================
# Install Octave language with control and signal packages
#=========================================================================================

if [[ "$action" == "octave" ]]
then
    if [[ "$platform" == "colab" ]]
    then
	apt install octave liboctave-dev
	octave --eval "pkg install -forge control signal"
    elif [[ "$platform" == "lawrencium" ]]
    then
	# Install octave from source
	module load gcc/6.3.0 lapack/3.8.0-gcc fftw/3.3.6-gcc
	export LD_PRELOAD=/global/software/sl-7.x86_64/modules/langs/gcc/6.3.0/lib64/libstdc++.so.6
	cd $HOME && wget --no-check-certificate https://ftpmirror.gnu.org/octave/octave-5.2.0.tar.gz
	tar -xzf octave-5.2.0.tar.gz && rm octave-5.2.0.tar.gz && cd octave-5.2.0
	mkdir -p .build $HOME/octave && cd .build && ./../configure --prefix=$HOME/octave
	make -j2
	make check
	make install
	rm -rf $HOME/octave-5.2.0
	# Install control package from source
	cd $HOME && wget https://phoenixnap.dl.sourceforge.net/project/octave/Octave%20Forge%20Packages/Individual%20Package%20Releases/control-3.2.0.tar.gz
	tar -xzf control-3.2.0.tar.gz && rm control-3.2.0.tar.gz && cd control-3.2.0/src
	tar -zxf slicot.tar.gz
	sed -i 's/DGEGS/DGGES/g' slicot/src/SG03AD.f slicot/src/SG03BD.f
	sed -i 's/DLATZM/DORMRZ/g' slicot/src/AB08NX.f slicot/src/AG08BY.f slicot/src/SB01BY.f slicot/src/SB01FY.f
	sed -i '/tar -xzf slicot.tar.gz/d' Makefile 
	cd $HOME && tar -zcvf control-3.2.0.tar.gz control-3.2.0/ && rm -rf control-3.2.0/
	$HOME/octave/bin/octave --eval "pkg install control-3.2.0.tar.gz"
	$HOME/octave/bin/octave --eval "pkg install -forge signal"
	rm $HOME/control-3.2.0.tar.gz
    fi
fi

#=========================================================================================
# Calculate probability map over large set of DSI files
#=========================================================================================

if [[ "$action" == "probmap" ]]
then
    module load python/3.6
    export PYTHONPATH=$SCRATCH/myenv/lib/python3.6/site-packages/:$PYTHONPATH
    python $MLDAS/bin/quickrun.py probmap -i $data -o $out
fi

#=========================================================================================
# Create weighted version of DSI files
#=========================================================================================

if [[ "$action" == "weight" ]]
then
    module load python/3.6
    export PYTHONPATH=$SCRATCH/myenv/lib/python3.6/site-packages/:$PYTHONPATH
    python $MLDAS/bin/quickrun.py weighting -i $data $weight -o $out -f $extra
fi

#========================================================================================
# Convert bulk of DSI files to HDF5 using MATLAB script
#========================================================================================

if [[ "$action" == "convert" ]]
then
    if [[ "$platform" != "lawrencium" ]]
    then
	echo "MAT-to-HDF5 converter only implemented on Lawrencium. Abort."
	exit
    fi
    module load matlab/r2017b
    out=$(realpath $out)
    mkdir -p ${out} && cd $MLDAS/matlab
    if [[ "$data" == *"*"* ]]
    then
	sed "s,mat_path,'${data}',g" SCRIPT_convert_dsi2hdf5.m > SCRIPT_convert_dsi2hdf5_tmp.m
	sed -i "s,h5_path,'${out}',g" SCRIPT_convert_dsi2hdf5_tmp.m
	matlab -nodisplay -nosplash -nodesktop -r SCRIPT_convert_dsi2hdf5_tmp
	rm SCRIPT_convert_dsi2hdf5_tmp.m
    else
        routine=$(basename -- $data)
        routine=${routine/.mat/}
	sed "s,mat_path,'${data}',g" SCRIPT_convert_dsi2hdf5.m > $routine.m
	sed -i "s,h5_path,'${out}',g" $routine.m
	matlab -nodisplay -nosplash -nodesktop -r $routine
	rm $routine.m
    fi
fi

#========================================================================================
# Perform cross-correlation of DSI or HDF5 files
#========================================================================================

make_config() {
    echo "[parameter]
dt = 0.002                    ;sample freq of original data
dt_new = 0.008                ;for iir/resample/...
butter_order = 3              ;for iir filter
winLen_sec = 0.5              ;for move mean
z = 0, 0.5, 1.0, 1.0, 0.5, 0  ;for interp1
F1 = 0
F2 = ${freqs[0]}
F3 = ${freqs[1]}
F4 = ${freqs[2]}
F5 = ${freqs[3]}
eCoeff = 1                                     ;for whitening 
master_index   = 0                             ;master channel
input_file     = $h5input
output_file    = $out/${fname}_corr.h5
input_dataset  = data
output_dataset = xcorr
view_enable_flag = false  ; true/false
view_start =     0,500
view_count = 30000,100" > $out/${fname}.config
}

if [[ "$action" == "xcorr" ]]
then
    module load gcc/7.4.0 hdf5/1.10.5-gcc-p fftw boost
    mkdir -p $out
    for input in ${data}
    do
	echo $input
	fname=$(basename -- $input)
	fname=${fname%%.*}
	if [[ "${input: -2}" == 'h5' ]]
	then
	    h5input=$input
	    make_config
	elif [[ "${input: -3}" == 'mat' ]]
	then
	    module load python/3.6
	    export PYTHONPATH=$SCRATCH/myenv/lib/python3.6/site-packages/:$PYTHONPATH
	    python $MLDAS/bin/quickrun.py xcorr_convert -i $input -o $out -f pre
	    h5input=$out/$fname.h5
	    make_config
	else
	    echo "File extension not recognized, must be .h5 or .mat. Abort"
	    exit
	fi
	if [[ "$software" == "dassa" ]]
	then
	    $HOME/dassa/xcorrelation -c $out/$config > $out/$fname.out
	elif [[ "$software" == "arrayudf" ]]
	then
            $HOME/ArrayUDF/examples/das/das-fft-full -c $out/${fname}.config > $out/$fname.out
	else
	    echo "Software not specified. Abort."
	    exit
	fi
	if [[ "${input: -3}" == 'mat' ]]
	then
	    python $MLDAS/bin/quickrun.py xcorr_convert -i $input -o $out -f post
	    rm $out/$fname.h5 $out/${fname}_corr.h5
	fi
	rm $out/$fname.config $out/$fname.out
    done
fi

#========================================================================================
# Perform incremented phase-weighted stack using either HDF5 or DSI files
#========================================================================================

if [[ "$action" == "pws" ]]
then
    if [[ "$software" == "matlab" ]]
    then
	module load gcc/6.3.0 lapack/3.8.0-gcc fftw/3.3.6-gcc
	export LD_PRELOAD=/global/software/sl-7.x86_64/modules/langs/gcc/6.3.0/lib64/libstdc++.so.6
	matfiles=(${data}/*.mat)
	num="${#matfiles[@]}"
	if [[ -z "$count" ]]
	then
	    for ((x=1; x<=$num; x++));
	    do
		$MLDAS/scripts/mldas.sh $input_args -c $x
	    done
	else
	    if (( $count > $num ))
	    then
		echo "Count ($count) must be less than or equal to the total number of HDF5 files in $data ($num). Abort."
		exit
	    fi
	    echo "Stacking $count cross-correlation spectra..."
	    mkdir -p ${out}/xcorr$count ${out}/stack$count
	    for ((x=0; x<$count; x++));
	    do
		ln -s ${matfiles[x]} ${out}/xcorr$count
	    done
            cd $MLDAS/matlab
	    if [[ -z "$weight" ]]
	    then
		cp SCRIPT_run_Stacking.m SCRIPT_run_Stacking_$count.m
	    else
		sed -i "s,MLweightFile,'${weight}',g" SCRIPT_run_Stacking_MLweights.m > SCRIPT_run_Stacking_$count.m
	    fi
	    sed -i "s,xcorr2stack,${out}/xcorr$count,g" SCRIPT_run_Stacking_$count.m
	    sed -i "s,stack_files,${out}/stack$count,g" SCRIPT_run_Stacking_$count.m
	    idx=$(printf "%05d" $count)
	    $HOME/octave/bin/octave -W SCRIPT_run_Stacking_$count.m 
	    mv ${out}/stack$count/Dsi_mstack.mat ${out}/Dsi_mstack_nstack$idx.mat
	    mv ${out}/stack$count/Dsi_pwstack.mat ${out}/Dsi_pwstack_nstack$idx.mat
	    rm -rf SCRIPT_run_Stacking_$count.m ${out}/xcorr$count ${out}/stack$count
	fi
    elif [[ "$software" == "dassa" ]]
    then
	module load gcc/7.4.0 hdf5/1.10.5-gcc-p fftw boost
	h5files=(${data}/*.h5)
	num="${#h5files[@]}"
	if [[ -z "$count" ]]
	then
	    for ((x=1; x<=$num; x++));
	    do
		$MLDAS/scripts/mldas.sh $input_args -c $x
	    done
	else
	    if (( $count > $num ))
	    then
		echo "Count ($count) must be less than or equal to the total number of HDF5 files in $data ($num). Abort."
		exit
	    fi
	    echo "Stacking $count cross-correlation spectra..."
	    mkdir -p ${out}/stack$count && cd ${out}/stack$count
	    cp $HOME/dassa/stack.config .
	    sed -i "s,/xcoor,/xcorr,g" stack.config
	    if [[ -z "$weight" ]]
	    then
		mkdir -p ${out}/xcorr$count
		for ((x=1; x<=$count; x++));
		do
		    ln -s ${h5files[x]} ${out}/xcorr$count
		done
		sed -i "s,/Users/dbin/work/arrayudf-git-svn-test-on-bitbucket/examples/das/stacking_files/xcorr_examples_h5,${out}/xcorr$count,g" stack.config
		sed -i "s,is_ml_weight = true,is_ml_weight = false,g" stack.config
	    else
		head -$count $weight > probmaps.txt
		#cp $weight probmaps.txt
		sed -i "s,/Users/dbin/work/arrayudf-git-svn-test-on-bitbucket/examples/das/stacking_files/xcorr_examples_h5,${data},g" stack.config
		sed -i "s,stack_ml_wight_sorted_chronologically.txt,probmaps.txt,g" stack.config
		sed -i "s,n_weighted_to_stack = -1,n_weighted_to_stack = $count,g" stack.config
		if [[ -z "$order" ]]
		then
		    sed -i "s,is_ml_weight_ordered = false;,is_ml_weight_ordered = false ;,g" stack.config
		else
		    sed -i "s,is_ml_weight_ordered = false;,is_ml_weight_ordered = true ;,g" stack.config
		fi
	    fi
	    if [[ "$data" == *"30min"* ]]
	    then
		sed -i "s,-59.9920000000000,-1859.992,g" stack.config
		sed -i "s,59.9920000000000,1859.992,g" stack.config
		sed -i "s,14999,464999,g" stack.config
		sed -i "s,201,501,g" stack.config
	    fi		
	    idx=$(printf "%05d" $count)
	    $HOME/dassa/stack -c stack.config
	    mv xcorr_examples_h5_stack_data_in_sum.h5         $out/xcorr_stack_data_in_sum_$idx.h5
	    mv xcorr_examples_h5_stack_final_pwstack.h5       $out/xcorr_stack_final_pwstack_$idx.h5
	    mv xcorr_examples_h5_stack_phaseWeight.h5         $out/xcorr_stack_phaseWeight_$idx.h5
	    mv xcorr_examples_h5_stack_semblanceWeight.h5     $out/xcorr_stack_semblanceWeight_$idx.h5
	    mv xcorr_examples_h5_stack_semblance_denom_sum.h5 $out/xcorr_stack_semblance_denom_sum_$idx.h5
	    cd ${out} && rm -rf xcorr$count stack$count
	fi
    else
	echo "Software not specified. Abort."
	exit
    fi
fi

#========================================================================================
# Do RMSD calculations (only available through MATLAB)
#========================================================================================

if [[ "$action" == "rmsd" ]]
then
    module load matlab/r2017b
    cd $MLDAS/matlab
    sed "s,stack_files/,${data}/,g" SCRIPT_calculate_rmsd_stack.m > script.m
    matlab -nodisplay -nosplash -nodesktop -r script
    mv spec_rmsd_pwstack.mat ${data}
    rm script.m
fi
