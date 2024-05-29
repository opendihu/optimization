
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <petsc.h>
int main(int argc, char* argv[]) {
   PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
   printf("MPI version %d.%d\n", MPI_VERSION, MPI_SUBVERSION);
   printf("Petsc version %d.%d.%d\n", PETSC_VERSION_MAJOR,PETSC_VERSION_MINOR,PETSC_VERSION_SUBMINOR);
   PetscFinalize();
   return EXIT_SUCCESS;
}

