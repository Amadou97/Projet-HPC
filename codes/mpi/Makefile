CC = mpicc
CFLAGS = -std=c99 -g -Wall -Wextra -Werror -O3 -march=native -fopenmp -I.
LDFLAGS =-fopenmp

# Uncomment these for OpenMP
#CFLAGS += -fopenmp
#LDFLAGS += -fopenmp

all: lanczos_modp checker_modp
lanczos_modp: mmio.o lanczos_modp.o
lanczos_modp.o: lanczos_modp.c mmio.h
checker_modp:   mmio.o checker_modp.o
checker_modp.o: checker_modp.c mmio.h

clean:
	rm -f *.o
	rm -f lanczos_modp checker_modp
	rm -f checkpoint_vector_p checkpoint_vector_v
