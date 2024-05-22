AMD_ARCH=gfx90a
ROCM_VER=6.1.0
ROCM_PATH ?=/opt/rocm-${ROCM_VER}
CC = ${ROCM_PATH}/bin/hipcc
FFLAGS = -DLITE_LOOP -DOMP_DEVICE  -fopenmp -ffree-form -O3 -g --offload-arch=gfx90a -DNDEBUG -fPIC
CFLAGS = -std=c++17 --offload-arch=${AMD_ARCH} -g -O3 -x hip -I${ROCM_PATH}/include
FFLAGS2 = -fopenmp --offload-arch=gfx90a
LDFLAGS= -L${ROCM_PATH}/lib -lamdhip64 -lroctx64 -lroctracer64 -lstdc++
EXAMPLES = repro_hip
FC=flang-new

all: $(EXAMPLES)

%.o: %.F90
	$(FC) $(FFLAGS) -o $@ -c $<

%.o: %.cpp
	$(CC) $(CFLAGS) -o $@ -c $<

repro_hip: repro_hip.o hip.o
	$(FC) $(FFLAGS2) hip.o repro_hip.o -o $@ $(LDFLAGS)

clean:
	rm -f ${EXAMPLES} *.o *~ *unknown* *amdgcn* *.d* *mod
