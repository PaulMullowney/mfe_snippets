#!/bin/bash -l
#
# This script can be used to single runs, groups of runs, profiles with rocprof, or profiles with omniperf.
# To use, copy this into the build directory and run via:
# ./script.sh -exe=VERSION -offload=OFFLOAD -cce=CCE_VERSION -arg2=ARG2 -case=CASE -cray_acc_debug=CRAY_ACC_DEBUG -oprof -rprof
#
# -exe=VERSION: LITE_LOOP, LITE_LOOP_HIP, LITE_LOOP_REVERSED, LITE_LOOP_REVERSED_HIP, LITE_LOOP_REVERSED_HOIST
# -offload=OFFLOAD: openmp-offload, hip, openmpi-offload, hip, openmp-offload
#          These match the version strings above
# -cce=CCE: 16.0.0 or whatever you build with. This variable is used for loading the right modules and creating profile names
# -arg2=ARG2 (optional): y dimension of the threadblock for LITE_LOOP_REVERSED_HIP.
#            When using OpenMP offload, this should be 64, 128, 256, ... (i.e. the size of the 2nd loop or the thread blocks size in the generated kernel).
# -case=CASE (optional): depends on the VERSION, OFFLOAD. Read the code below. This is useful for running single cases of each exe.
# -cray_acc_debug=CRAY_ACC_DEBUG (optional): useful for seeing how OpenMP kernels are launched. Use -case argument with an -offload=openmp-offload
# -oprof (optional): run omniperf on a single case or all cases for VERSION/OFFLOAD
# -rprof (optional): run rocprof on a single case or all cases for VERSION/OFFLOAD
#
#
oprofile=
rprofile=
cray_acc_debug=0
arg2=

for i in "$@"; do
    case "$1" in
        -cce=*|--cce=*)
            cce="${i#*=}"
            shift # past argument=value
            ;;
        -offload=*|--offload=*)
            offload="${i#*=}"
            shift # past argument=value
            ;;
        -exe=*|--exe=*)
            exe="${i#*=}"
            shift # past argument=value
            ;;
        -arg2=*|--arg2=*)
            arg2="${i#*=}"
            shift # past argument=value
            ;;
        -case=*|--case=*)
            case="${i#*=}"
            shift # past argument=value
            ;;
        -cray_acc_debug=*|--cray_acc_debug=*)
            cray_acc_debug="${i#*=}"
            shift # past argument=value
            ;;
        -oprof|--oprof)
            oprofile=YES
            shift # past argument=value
            ;;
        -rprof|--rprof)
            rprofile=YES
            shift # past argument=value
            ;;
        --)
            shift
            break
            ;;
    esac
done


module load PrgEnv-cray
if [[ $cce == "16.0.1" ]]
then
	 module load cpe/23.09
else
	 module load cpe/23.05
	 #	 GCC_X86_64=/opt/cray/pe/gcc/12.2.0/snos
fi
module load rocm/5.4.3
module load craype-accel-amd-gfx90a
module unload darshan-runtime

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

if [[ $oprofile ]]
then
	 source ~/crusher-venv/bin/activate
	 export OMNIPERF_DIR=/ccs/home/mullowne/omniperf-1.0.10
	 export PATH=$OMNIPERF_DIR/1.0.10/bin:$PATH
	 export PYTHONPATH=$OMNIPERF_DIR/python-libs
	 echo $PYTHONPATH
fi

if [[ $exe == "LITE_LOOP" ]]
then
	 cases=( 1 2 3 4 5 6 7 8 )
elif [[ $exe == "LITE_LOOP_REVERSED" ]]
then
	 cases=( 1 2 3 4 5 )
elif [[ $exe == "LITE_LOOP_REVERSED_HOIST" ]]
then
	 cases=( 1 2 3 )
elif [[ $exe == "LITE_LOOP_HIP" ]]
then
	 cases=( 64 128 256 )
elif [[ $exe == "LITE_LOOP_REVERSED_HIP" ]]
then
	 cases=( 64 )
fi

EXE=driver_${exe}_${offload}
echo ${EXE}

if [[ $case ]]
then
    if [[ $rprofile ]]
	 then
		  if [[ $arg2 ]]
		  then
				srun -N 1 -n 1 --gpus-per-node=8 --gpu-bind=closest rocprof --roctx-trace --hip-trace --stats -o rocprofiles/${exe}_${case}_${arg2}_${cce}.csv ${EXE} ${case} ${arg2}
		  else
				srun -N 1 -n 1 --gpus-per-node=8 --gpu-bind=closest rocprof --roctx-trace --hip-trace --stats -o rocprofiles/${exe}_${case}_${cce}.csv ${EXE} ${case} ${arg2}
		  fi
	 else
		  CRAY_ACC_DEBUG=$cray_acc_debug srun -N 1 -n 1 --gpus-per-node=8 --gpu-bind=closest ${EXE} ${case} ${arg2}
	 fi
else
for i in "${cases[@]}"
do
	 if [[ $oprofile ]]
	 then
		  srun -N 1 -n 1 --gpus-per-node=8 --gpu-bind=closest omniperf profile -n temp -- ${EXE} ${i} ${arg2}
		  if [[ $arg2 ]]
		  then
				mv workloads/temp workloads/${exe}_${i}_${arg2}_${cce}
		  else
				mv workloads/temp workloads/${exe}_${i}_${cce}
		  fi
	 elif [[ $rprofile ]]
	 then
		  if [[ $arg2 ]]
		  then
				srun -N 1 -n 1 --gpus-per-node=8 --gpu-bind=closest rocprof --roctx-trace --hip-trace --stats -o rocprofiles/${exe}_${i}_${arg2}_${cce}.csv ${EXE} ${i} ${arg2}
		  else
				srun -N 1 -n 1 --gpus-per-node=8 --gpu-bind=closest rocprof --roctx-trace --hip-trace --stats -o rocprofiles/${exe}_${i}_${cce}.csv ${EXE} ${i} ${arg2}
		  fi
	 else
		  srun -N 1 -n 1 --gpus-per-node=8 --gpu-bind=closest ${EXE} ${i} ${arg2}
	 fi
done
fi
