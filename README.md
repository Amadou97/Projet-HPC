# Projet-HPC
Parallélisation de  l'algorithme de block-Lanczos modulo p


Le projet contient trois dossiers :
1-mpi: contient le code parallel avec mpi uniquement
2-OpenMP : contient le code parallel avec OpenMP uniquement
3-mpi+OpenMP : contient le code parallel avec mpi et OpenMP combine

NB : Dans chaque dossier y trouve un makefile pour compiler


connection : ssh abayo@access.grid5000.fr 
             ssh nancy puis OAR

transfert de fichier : scp hello.c abayo@access.grid5000.fr:nancy/


Reservation de ressoursces:
On reserve un host pendant 1 heure : oarsub -I -l /host=1,walltime=1

On reserve 4 coeurs pendant 10 heures : arsub -I -l /core=4,walltime=10

Code mpi :
execution sur 2 processeurs : mpiexec --n 2 lanczos_modp --matrix TF16/TF16.mtx --prime 65537 --right --n 2 --output kernel.mtx

Code OpenM
execution sur 2 threads : OMP_NUM_THREADS=2 ./lanczos_modp --matrix TF16/TF16.mtx --prime 65537 --right --n 2 --output kernel.mtx

code mpi+OpenMP:
commande de reservation : oarsub -I -l /host=8/cpu=1/core=16,walltime=1
mpirun --hostfile $OAR_NODEFILE --n 8  --bind-to core -x OMP_NUM_THREADS=1  ./lanczos_modp --matrix ../matrice/TF18/TF18.mtx --prime 65537 --right --n 2 --output kernel.mtx


Test du checkpointing :
Pour utiliser l'option de checkpoint,il suffit de couper l'option de l'excution de n'importe qu'elle des version par exemple en faisant ctr-C.Ensuite il suffit de 
relancer la même commande avec en plus l'option --checkpoint. 
OpenMP:
$ export OMP_NUM_THREADS=4
$ ./lanczos_modp --matrix TF16/TF17.mtx --prime 65537 --right --n 2 --output kernel.mtx //Ctr-C après 200 itérations
$ ./lanczos_modp --matrix TF16/TF17.mtx --checkpoint --prime 65537 --right --n 2 --output kernel.mtx

Mpi:
$ mpiexec --n 2 lanczos_modp --matrix TF16/TF17.mtx --prime 65537 --right --n 2 --output kernel.mtx //Ctr-c après 800 itérations
$ mpiexec --n 2 lanczos_modp --checkpoint --matrix TF16/TF17.mtx --prime 65537 --right --n 2 --output kernel.mtx

Mpi+OpenMP:
$ mpirun --hostfile $OAR_NODEFILE --n 4  --bind-to core -x OMP_NUM_THREADS=2  ./lanczos_modp --matrix ../matrice/TF18/TF18.mtx --prime 65537 --right --n 2 --output kernel.mtx
 //Ctr-c après 800 itérations
$ mpirun --hostfile $OAR_NODEFILE --n 4  --bind-to core -x OMP_NUM_THREADS=2  ./lanczos_modp --checkpoint--matrix ../matrice/TF18/TF18.mtx --prime 65537 --right --n 2 --output kernel.mtx
