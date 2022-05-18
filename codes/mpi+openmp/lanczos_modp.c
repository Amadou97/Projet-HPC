/* 
 * Sequential implementation of the Block-Lanczos algorithm.
 *
 * This is based on the paper: 
 *     "A modified block Lanczos algorithm with fewer vectors" 
 *
 *  by Emmanuel Thomé, available online at 
 *      https://hal.inria.fr/hal-01293351/document
 *
 * Authors : Charles Bouillaguet
 *
 * v1.00 (2022-01-18)
 *
 * USAGE: 
 *      $ ./lanczos_modp --prime 65537 --n 4 --matrix random_small.mtx
 *
 */
#define _POSIX_C_SOURCE  1  // ctime
#include <inttypes.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <err.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <assert.h>
#include <omp.h>
#include <mpi.h>
#include <mmio.h>

int rang;
int proc;
MPI_Status status;
int tag = 0;


typedef uint64_t u64;
typedef uint32_t u32;

/******************* global variables ********************/

long n = 1;
u64 prime;
char *matrix_filename;
char *kernel_filename;
bool right_kernel = false;
int stop_after = -1;
bool checkpoint=false;

int n_iterations;      /* variables of the "verbosity engine" */
double start;
double last_print;
bool ETA_flag;
int expected_iterations;

/******************* sparse matrix data structure **************/

struct sparsematrix_t {
        int nrows;        // dimensions
        int ncols;
        long int nnz;     // number of non-zero coefficients
        int *i;           // row indices (for COO matrices)
        int *j;           // column indices
        u32 *x;           // coefficients
};

/******************* openmp custom reduction *********************/
void sum_mod_func(u32 * out,u32 * in){
    (*out)=((*out)+(*in))%prime;
}
#pragma omp declare reduction(sum_mod:u32:sum_mod_func(&omp_out,&omp_in)) \
                    initializer( omp_priv = 0 )

/******************* pseudo-random generator (xoshiro256+) ********************/

/* fixed seed --- this is bad */
u64 rng_state[4] = {0x1415926535, 0x8979323846, 0x2643383279, 0x5028841971};

u64 rotl(u64 x, int k)
{
        u64 foo = x << k;
        u64 bar = x >> (64 - k);
        return foo ^ bar;
}

u64 random64()
{
        u64 result = rotl(rng_state[0] + rng_state[3], 23) + rng_state[0];
        u64 t = rng_state[1] << 17;
        rng_state[2] ^= rng_state[0];
        rng_state[3] ^= rng_state[1];
        rng_state[1] ^= rng_state[2];
        rng_state[0] ^= rng_state[3];
        rng_state[2] ^= t;
        rng_state[3] = rotl(rng_state[3], 45);
        return result;
}

/******************* utility functions ********************/

double wtime()
{
        struct timeval ts;
        gettimeofday(&ts, NULL);
        return (double) ts.tv_sec + ts.tv_usec / 1e6;
}

/* represent n in <= 6 char  */
void human_format(char * target, long n) {
        if (n < 1000) {
                sprintf(target, "%" PRId64, n);
                return;
        }
        if (n < 1000000) {
                sprintf(target, "%.1fK", n / 1e3);
                return;
        }
        if (n < 1000000000) {
                sprintf(target, "%.1fM", n / 1e6);
                return;
        }
        if (n < 1000000000000ll) {
                sprintf(target, "%.1fG", n / 1e9);
                return;
        }
        if (n < 1000000000000000ll) {
                sprintf(target, "%.1fT", n / 1e12);
                return;
        }
}

/************************** command-line options ****************************/

void usage(char ** argv)
{
        printf("%s [OPTIONS]\n\n", argv[0]);
        printf("Options:\n");
        printf("--matrix FILENAME           MatrixMarket file containing the spasre matrix\n");
        printf("--prime P                   compute modulo P\n");
        printf("--n N                       blocking factor [default 1]\n");
        printf("--output-file FILENAME      store the block of kernel vectors\n");
        printf("--right                     compute right kernel vectors\n");
        printf("--left                      compute left kernel vectors [default]\n");
        printf("--stop-after N              stop the algorithm after N iterations\n");
        printf("\n");
        printf("The --matrix and --prime arguments are required\n");
        printf("The --stop-after and --output-file arguments mutually exclusive\n");
        exit(0);
}

void process_command_line_options(int argc, char ** argv)
{
        struct option longopts[8] = {
                {"matrix", required_argument, NULL, 'm'},
                {"prime", required_argument, NULL, 'p'},
                {"n", required_argument, NULL, 'n'},
                {"output-file", required_argument, NULL, 'o'},
                {"right", no_argument, NULL, 'r'},
                {"left", no_argument, NULL, 'l'},
                {"stop-after", required_argument, NULL, 's'},
                {NULL, 0, NULL, 0}
        };
        char ch;
        while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
                switch (ch) {
                case 'm':
                        matrix_filename = optarg;
                        break;
                case 'n':
                        n = atoi(optarg);
                        break;
                case 'p':
                        prime = atoll(optarg);
                        break;
                case 'o':
                        kernel_filename = optarg;
                        break;
                case 'r':
                        right_kernel = true;
                        break;
                case 'l':
                        right_kernel = false;
                        break;
                case 's':
                        stop_after = atoll(optarg);
                        break;
                default:
                        errx(1, "Unknown option\n");
                }
        }

        /* missing required args? */
        if (matrix_filename == NULL || prime == 0)
                usage(argv);
        /* exclusive arguments? */
        if (kernel_filename != NULL && stop_after > 0)
                usage(argv);
        /* range checking */
        if (prime > 0x3fffffdd) {
                errx(1, "p is capped at 2**30 - 35.  Slighly larger values could work, with the\n");
                printf("suitable code modifications.\n");
                exit(1);
        }
}
/***************  checkpoint  *******************/
void store_vector(u32 * v,char * str,int size,int nb_iteration){
    FILE *fptr;
    fptr=fopen(str,"w+");
    if(fptr==NULL){
        perror("failed fopen");
        exit(3);
        }
    fprintf(fptr,"%d\n",nb_iteration);
    fprintf(fptr,"%d\n",size);
    for(int i=0;i<size;i++){
        fprintf(fptr,"%d\n",v[i]);
        }
    
    fclose(fptr);
    }
void store_checkpoint(u32 * v,u32 * p,int size,int nb_iteration){
    store_vector(v,"checkpoint_vector_v",size,nb_iteration);
    store_vector(p,"checkpoint_vector_p",size,nb_iteration);
    }
u32 * load_vector(char * str,int * nb_iteration){
    FILE *fptr;
    u32 *v=NULL;
    int size;
    int ret;
    fptr=fopen(str,"r+");
    if(fptr==NULL){
        perror("failed fopen");
        exit(3);
        }
    ret=fscanf(fptr,"%d\n",nb_iteration);
    ret=fscanf(fptr,"%d\n",&size);
    
    if(ret==0){
        perror("failed fscanf");
        exit(5);
        }
    
    v=malloc(sizeof(*v)*size);
    if(v==NULL){
        perror("failed malloc");
        exit(4);
        }
    for(int i=0;i<size;i++){
        ret=fscanf(fptr,"%d\n",&(v[i]));
        }
    
    fclose(fptr);
    return v;
    }
void load_checkpoint(u32 **v,u32 **p,int * nb_iteration){
    (*v)=load_vector("checkpoint_vector_v",nb_iteration);
    (*p)=load_vector("checkpoint_vector_p",nb_iteration);
    }
/****************** sparse matrix operations ******************/

/* Load a matrix from a file in "list of triplet" representation */
void sparsematrix_mm_load(struct sparsematrix_t * M, char const * filename)
{
        int nrows = 0;
        int ncols = 0;
        long nnz = 0;

        printf("Loading matrix from %s\n", filename);
        fflush(stdout);

        FILE *f = fopen(filename, "r");
        if (f == NULL)
                err(1, "impossible d'ouvrir %s", filename);

        /* read the header, check format */
        MM_typecode matcode;
        if (mm_read_banner(f, &matcode) != 0)
                errx(1, "Could not process Matrix Market banner.\n");
        if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode))
                errx(1, "Matrix Market type: [%s] not supported (only sparse matrices are OK)", 
                        mm_typecode_to_str(matcode));
        if (!mm_is_general(matcode) || !mm_is_integer(matcode))
                errx(1, "Matrix type [%s] not supported (only integer general are OK)", 
                        mm_typecode_to_str(matcode));
        if (mm_read_mtx_crd_size(f, &nrows, &ncols, &nnz) != 0)
                errx(1, "Cannot read matrix size");
        fprintf(stderr, "  - [%s] %d x %d with %ld nz\n", mm_typecode_to_str(matcode), nrows, ncols, nnz);
        fprintf(stderr, "  - Allocating %.1f MByte\n", 1e-6 * (12.0 * nnz));

        /* Allocate memory for the matrix */
        int *Mi = malloc(nnz * sizeof(*Mi));
        int *Mj = malloc(nnz * sizeof(*Mj));
        u32 *Mx = malloc(nnz * sizeof(*Mx));
        if (Mi == NULL || Mj == NULL || Mx == NULL)
                err(1, "Cannot allocate sparse matrix");

        /* Parse and load actual entries */
        double start = wtime();
        for (long u = 0; u < nnz; u++) {
                int i, j;
                u32 x;
                if (3 != fscanf(f, "%d %d %d\n", &i, &j, &x))
                        errx(1, "parse error entry %ld\n", u);
                Mi[u] = i - 1;  /* MatrixMarket is 1-based */
                Mj[u] = j - 1;
                Mx[u] = x % prime;
                
                // verbosity
                if ((u & 0xffff) == 0xffff) {
                        double elapsed = wtime() - start;
                        double percent = (100. * u) / nnz;
                        double rate = ftell(f) / 1048576. / elapsed;
                        printf("\r  - Reading %s: %.1f%% (%.1f MB/s)", matrix_filename, percent, rate);
                }
        }

        /* finalization */
        fclose(f);
        printf("\n");
        M->nrows = nrows;
        M->ncols = ncols;
        M->nnz = nnz;
        M->i = Mi;
        M->j = Mj;
        M->x = Mx;
}

void test_order(int * tab,long nnz){
        for(long i=0;i<nnz-1;i++){
                if(tab[tab[i]]>tab[i+1]){
                        printf("non trié\n");
                        return;
                        }
                }
        printf("trié\n");
        }


/* y += M*x or y += transpose(M)*x, according to the transpose flag */ 
void sparse_matrix_vector_product(u32 * y, struct sparsematrix_t const * M, u32 const * x, bool transpose)
{
        long nnz = M->nnz;
        int nrows = transpose ? M->ncols : M->nrows;
        //int const * Mi = M->i;
        //int const * Mj = M->j;
        int *Mi = transpose ? M->j : M->i;
        int *Mj = transpose ? M->i : M->j;
        u32 const * Mx = M->x;
        
        for (long i = 0; i < nrows * n; i++){
                y[i] = 0;
        }
        //l'important c'est que les i soit triées
        #pragma omp parallel for reduction(sum_mod:y[0:nrows*n])
        for (long k = 0; k < nnz ; k++) {
                //int i = transpose ? Mj[k] : Mi[k];
                //int j = transpose ? Mi[k] : Mj[k];
                int i=Mi[k];
                int j=Mj[k];
                u64 v = Mx[k];
                
                for (int l = 0; l < n; l++) {
                        u64 a = y[i * n + l ];
                        u64 b = x[j * n + l ];
                        y[i * n + l] = (a + v * b) % prime;
                }
        }
}

/****************** dense linear algebra modulo p *************************/ 

/* C += A*B   for n x n matrices */
void matmul_CpAB(u32 * C, u32 const * A, u32 const * B)
{
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        for (int k = 0; k < n; k++) {
                                u64 x = C[i * n + j];
                                u64 y = A[i * n + k];
                                u64 z = B[k * n + j];
                                C[i * n + j] = (x + y * z) % prime;
                        }
}

/* C += transpose(A)*B   for n x n matrices */
void matmul_CpAtB(u32 * C, u32 const * A, u32 const * B)
{
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        for (int k = 0; k < n; k++) {
                                u64 x = C[i * n + j];
                                u64 y = A[k * n + i];
                                u64 z = B[k * n + j];
                                C[i * n + j] = (x + y * z) % prime;
                        }
}

/* return a^(-1) mod b */
u32 invmod(u32 a, u32 b)
{
        long int t = 0;  
        long int nt = 1;  
        long int r = b;  
        long int nr = a % b;
        while (nr != 0) {
                long int q = r / nr;
                long int tmp = nt;  
                nt = t - q*nt;  
                t = tmp;
                tmp = nr;  
                nr = r - q*nr;  
                r = tmp;
        }
        if (t < 0)
                t += b;
        return (u32) t;
}

/* 
 * Given an n x n matrix U, compute a "partial-inverse" V and a diagonal matrix
 * d such that d*V == V*d == V and d == V*U*d. Return the number of pivots.
 */ 
int semi_inverse(u32 const * M_, u32 * winv, u32 * d)
{
        u32 M[n * n];
        int npiv = 0;
        for (int i = 0; i < n * n; i++)   /* copy M <--- M_ */
                M[i] = M_[i];
        /* phase 1: compute d */
        for (int i = 0; i < n; i++)       /* setup d */
                d[i] = 0;
        for (int j = 0; j < n; j++) {     /* search a pivot on column j */
                int pivot = n;
                for (int i = j; i < n; i++)
                        if (M[i*n + j] != 0) {
                                pivot = i;
                                break;
                        }
                if (pivot >= n)
                        continue;         /* no pivot found */
                d[j] = 1;
                npiv += 1;
                u64 pinv = invmod(M[pivot*n + j], prime);  /* multiply pivot row to make pivot == 1 */
                for (int k = 0; k < n; k++) {
                        u64 x = M[pivot*n + k];
                        M[pivot*n + k] = (x * pinv) % prime;
                }
                for (int k = 0; k < n; k++) {   /* swap pivot row with row j */
                        u32 tmp = M[j*n + k];
                        M[j*n + k] = M[pivot*n + k];
                        M[pivot*n + k] = tmp;
                }
                for (int i = 0; i < n; i++) {  /* eliminate everything else on column j */
                        if (i == j)
                                continue;
                        u64 multiplier = M[i*n+j];
                        for (int k = 0; k < n; k++) {
                                u64 x = M[i * n + k];
                                u64 y = M[j * n + k];
                                M[i * n + k] = (x + (prime - multiplier) * y) % prime;  
                        }
                }
        }
        /* phase 2: compute d and winv */
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                        M[i*n + j] = (d[i] && d[j]) ? M_[i*n + j] : 0;
                        winv[i*n + j] = ((i == j) && d[i]) ? 1 : 0;
                }
        npiv = 0;
        for (int i = 0; i < n; i++)
                d[i] = 0;
        /* same dance */
        for (int j = 0; j < n; j++) { 
                int pivot = n;
                for (int i = j; i < n; i++)
                        if (M[i*n + j] != 0) {
                                pivot = i;
                                break;
                        }
                if (pivot >= n)
                        continue;
                d[j] = 1;
                npiv += 1;
                u64 pinv = invmod(M[pivot*n + j], prime);
                for (int k = 0; k < n; k++) {
                        u64 x = M[pivot*n + k];
                        M[pivot*n + k] = (x * pinv) % prime;
                }
                for (int k = 0; k < n; k++) {
                        u32 tmp = M[j*n + k];
                        M[j*n + k] = M[pivot*n + k];
                        M[pivot*n + k] = tmp;
                }
                for (int k = 0; k < n; k++) {
                        u64 x = winv[pivot * n + k];
                        winv[pivot * n + k] = (x * pinv) % prime;
                }
                for (int k = 0; k < n; k++) {
                        u32 tmp = winv[j * n + k];
                        winv[j * n + k] = winv[pivot * n + k];
                        winv[pivot * n + k] = tmp;
                }
                for (int i = 0; i < n; i++) {
                        if (i == j)
                                continue;
                        u64 multiplier = M[i * n + j];
                        for (int k = 0; k < n; k++) {
                                u64 x = M[i * n + k];
                                u64 y = M[j * n + k];
                                M[i * n + k] = (x + (prime - multiplier) * y) % prime;
                                u64 w = winv[i * n + k];
                                u64 z = winv[j * n + k];
                                winv[i * n + k] = (w + (prime - multiplier) * z) % prime;  
                        }
                }
        }
        return npiv;
}
/*************************** block-Lanczos algorithm ************************/

/* Computes vtAv <-- transpose(v) * Av, vtAAv <-- transpose(v) * Av */
void block_dot_products(u32 * vtAv, u32 * vtAAv, int N, u32 const * Av, u32 const * v)
{       
        for (int i = 0; i < n * n; i++){
                vtAv[i]  = 0;
                vtAAv[i] = 0;
        }
        
        #pragma omp parallel for reduction(sum_mod:vtAv[0:n*n])
        for (int i = 0; i < N; i += n){
                //printf("threads = %d\n",omp_get_num_threads());
                matmul_CpAtB(vtAv,   &v[i*n], &Av[i*n]);
        }
        
        #pragma omp parallel for reduction(sum_mod:vtAAv[0:n*n])
        for (int i = 0; i < N; i += n){
                matmul_CpAtB(vtAAv, &Av[i*n], &Av[i*n]);
        }
}

/* Compute the next values of v (in tmp) and p */
void orthogonalize(u32 * v, u32 * tmp, u32 * p, u32 * d, u32 const * vtAv, const u32 *vtAAv, 
        u32 const * winv, int N, u32 const * Av)
{
        /* compute the n x n matrix c */
        u32 c[n * n];
        u32 spliced[n * n];
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                        spliced[i*n + j] = d[j] ? vtAAv[i * n + j] : vtAv[i * n + j];
                        c[i * n + j] = 0;
                }
        matmul_CpAB(c, winv, spliced);
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        c[i * n + j] = prime - c[i * n + j];

        u32 vtAvd[n * n];
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        vtAvd[i*n + j] = d[j] ? prime - vtAv[i * n + j] : 0;

        /* compute the next value of v ; store it in tmp */        
        for (long i = 0; i < N; i++)
                for (long j = 0; j < n; j++)
                        tmp[i*n + j] = d[j] ? Av[i*n + j] : v[i * n + j];
        
        #pragma omp parallel for
        for (long i = 0; i < N; i += n)
                matmul_CpAB(&tmp[i*n], &v[i*n], c);
        
        #pragma omp parallel for
        for (long i = 0; i < N; i += n)
                matmul_CpAB(&tmp[i*n], &p[i*n], vtAvd);
        
        /* compute the next value of p */
        for (long i = 0; i < N; i++)
                for (long j = 0; j < n; j++)
                        p[i * n + j] = d[j] ? 0 : p[i * n + j];
        #pragma omp parallel for
        for (long i = 0; i < N; i += n)
                matmul_CpAB(&p[i*n], &v[i*n], winv);
}

void verbosity()
{
        double elapsed = wtime() - start;
        if (elapsed - last_print < 1) 
                return;

        last_print = elapsed;
        double per_iteration = elapsed / n_iterations;
        double estimated_length = expected_iterations * per_iteration;
        time_t end = start + estimated_length;
        if (!ETA_flag) {
                int d = estimated_length / 86400;
                estimated_length -= d * 86400;
                int h = estimated_length / 3600;
                estimated_length -= h * 3600;
                int m = estimated_length / 60;
                estimated_length -= m * 60;
                int s = estimated_length;
                printf("    - Expected duration : ");
                if (d > 0)
                        printf("%d j ", d);
                if (h > 0)
                        printf("%d h ", h);
                if (m > 0)
                        printf("%d min ", m);
                printf("%d s\n", s);
                ETA_flag = true;
        }
        char ETA[30];
        ctime_r(&end, ETA);
        ETA[strlen(ETA) - 1] = 0;  // élimine le \n final
        printf("\r    - iteration %d / %d. %.3fs per iteration. ETA: %s", 
                n_iterations, expected_iterations, per_iteration, ETA);
        fflush(stdout);
}

/* optional tests */
void correctness_tests(u32 const * vtAv, u32 const * vtAAv, u32 const * winv, u32 const * d)
{
        /* vtAv, vtAAv, winv are actually symmetric + winv and d match */
        for (int i = 0; i < n; i++) 
                for (int j = 0; j < n; j++) {
                        assert(vtAv[i*n + j] == vtAv[j*n + i]);
                        assert(vtAAv[i*n + j] == vtAAv[j*n + i]);
                        assert(winv[i*n + j] == winv[j*n + i]);
                        assert((winv[i*n + j] == 0) || d[i] || d[j]);
                }
        /* winv satisfies d == winv * vtAv*d */
        u32 vtAvd[n * n];
        u32 check[n * n];
        for (int i = 0; i < n; i++) 
                for (int j = 0; j < n; j++) {
                        vtAvd[i*n + j] = d[j] ? vtAv[i*n + j] : 0;
                        check[i*n + j] = 0;
                }
        matmul_CpAB(check, winv, vtAvd);
        for (int i = 0; i < n; i++) 
                for (int j = 0; j < n; j++)
                        if (i == j)
                                assert(check[j*n + j] == d[i]);
                        else
                                assert(check[i*n + j] == 0);
}
/* check that we actually computed a kernel vector */
void final_check(int nrows, int ncols, u32 const * v, u32 const * vtM)
{
        printf("Final check:\n");
        /* Check if v != 0 */
        bool good = false;
        for (long i = 0; i < nrows; i++)
                for (long j = 0; j < n; j++)
                        good |= (v[i*n + j] != 0);
        if (good)
                printf("  - OK:    v != 0\n");
        else
                printf("  - KO:    v == 0\n");
                
        /* tmp == Mt * v. Check if tmp == 0 */
        good = true;
        for (long i = 0; i < ncols; i++)
                for (long j = 0; j < n; j++)
                        good &= (vtM[i*n + j] == 0);
        if (good)
                printf("  - OK: vt*M == 0\n");
        else
                printf("  - KO: vt*M != 0\n");                
}

void sum_modulo(void* inputBuffer, void* outputBuffer, int* len, MPI_Datatype* datatype){
    u32* input = (u32*)inputBuffer;
    u32* output = (u32*)outputBuffer;
    datatype=datatype;
        for(int i = 0; i < *len; i++){
                u64 a = output[i];
                u64 b = input[i];
                output[i] = (a + b)%prime;
        }
}

/* Solve x*M == 0 or M*x == 0 (if transpose == True) */
u32 * block_lanczos(struct sparsematrix_t const * M, int n, bool transpose)
{
        if ( rang == 0 ){
                printf("Block Lanczos\n");
        }
        
        /************* preparations **************/
        
        /* allocate blocks of vectors */
        int nrows = transpose ? M->ncols : M->nrows;
        int ncols = transpose ? M->nrows : M->ncols;
        long block_size = nrows * n;
        long Npad = ((nrows + n - 1) / n) * n;
        long Mpad = ((ncols + n - 1) / n) * n;
        long block_size_pad = (Npad > Mpad ? Npad : Mpad) * n;
        char human_size[8];
        human_format(human_size, 4 * sizeof(int) * block_size_pad);
        if ( rang == 0){
                printf("  - Extra storage needed: %sB\n", human_size);
        }
        
        int nb_block=block_size_pad/(n*n);
        int block_proc=nb_block/proc;
        int line = block_proc*n*n;
        int line_reste  = 0;
        int gap_store=800;
        if ( rang == proc-1){
                line_reste = ( nb_block %proc)*n*n;
        }
        
        // Declare the counts
        int counts[proc];
        // Declare the displacements
        int displacements[proc];
        
        for(int i = 0; i < proc; i++){
                if( i == proc - 1){
                        displacements[i] = i*line;
                        counts[i] = line + (nb_block%proc)*n*n;
                } else {
                        displacements[i] = i*line;
                        counts[i] = line;
                }
        }
        printf("test\n");
        u32 *v,*p,*Av,*tmp;
        u32* vbis       = malloc(sizeof(*v) * block_size_pad);        
        u32 *tmpbis     = malloc(sizeof(*tmpbis) * block_size_pad);
        u32 *tmpbisbis  = malloc(sizeof(*tmpbis) * block_size_pad);
        u32* Avbisbis   = malloc(sizeof(*Av) * block_size_pad);
        u32* Avbis      = malloc(sizeof(*Av) * block_size_pad);
        Av=NULL;
        int nb_iteration_done=0;
        if(rang==0){
                if(checkpoint){
                        load_checkpoint(&v,&p,&nb_iteration_done);
                        }
                else{
                        v = malloc(sizeof(*v) * block_size_pad);
                        p = malloc(sizeof(*p) * block_size_pad);
                        }
                }
        else{
                v = malloc(sizeof(*v) * block_size_pad);
                p = malloc(sizeof(*p) * block_size_pad);
                }
        tmp = malloc(sizeof(*tmp) * block_size_pad);
        Av = malloc(sizeof(*Av) * block_size_pad);
        
        if (v == NULL || tmp == NULL || Av == NULL || p == NULL){
                errx(1, "impossible d'allouer les blocs de vecteur");
        }
        
        if(rang==0){
                expected_iterations = 1 + (ncols / n) -nb_iteration_done;
                char human_its[8];
                human_format(human_its, expected_iterations);
                printf("  - Expecting %s iterations\n", human_its);
                if(checkpoint){
                        for (long i = 0; i < block_size_pad; i++) {
                                tmp[i] = v[i];
                                Av[i]=0;
                                }
                        }
                else{
                        for (long i = 0; i < block_size_pad; i++) {
                                Av[i] = 0;
                                v[i] = 0;
                                p[i] = 0;
                                tmp[i] = v[i];
                                }
                        for (long i = 0; i < block_size; i++)
                                v[i] = random64() % prime;
                        }
                        
                }
        
        
        /************* main loop *************/
        if ( rang == 0){
                printf("  - Main loop\n");
        }

        MPI_Op operation;
        MPI_Op_create(&sum_modulo, 1, &operation);
        //MPI_Bcast(v,  block_size_pad, MPI_UINT32_T, 0,MPI_COMM_WORLD);
        if(rang==0){
                MPI_Scatterv( p , counts, displacements, MPI_UINT32_T,  MPI_IN_PLACE, line + line_reste, MPI_UINT32_T, 0, MPI_COMM_WORLD);
                }
        else{
                MPI_Scatterv( p , counts, displacements, MPI_UINT32_T,  p , line + line_reste, MPI_UINT32_T, 0, MPI_COMM_WORLD);
                }
        
        for (long i = 0; i < block_size_pad; i++) {
                tmp[i] = v[i];
                }
        start = wtime();
        bool stop = false;
        while (true) {
                if (stop_after > 0 && n_iterations == stop_after)
                        break;
                
                if(rang==0){
                        if(n_iterations%gap_store==gap_store-1){
                                MPI_Gatherv(MPI_IN_PLACE,line+line_reste,MPI_UINT32_T,p,counts,displacements,MPI_UINT32_T,0,MPI_COMM_WORLD);
                                store_checkpoint(v,p,block_size_pad,n_iterations);
                                }
                        }
                else{
                        if(n_iterations%gap_store==gap_store-1){
                                MPI_Gatherv(p,line+line_reste,MPI_UINT32_T,NULL,NULL,NULL,MPI_UINT32_T,0,MPI_COMM_WORLD);
                                }
                        }
                MPI_Bcast(v,  block_size_pad, MPI_UINT32_T, 0,MPI_COMM_WORLD);
                sparse_matrix_vector_product(tmpbisbis, M, v,  !transpose);
                for (long i = 0; i < block_size_pad; i++) {
                        tmp[i] = 0;
                        Av[i]  = 0;
                }
                
                MPI_Allreduce(tmpbisbis, tmp, block_size_pad, MPI_UINT32_T, operation, MPI_COMM_WORLD);
                sparse_matrix_vector_product(Avbisbis,  M, tmp, transpose);
                MPI_Reduce(Avbisbis, Av, block_size_pad, MPI_UINT32_T, operation, 0, MPI_COMM_WORLD);

                u32 vtAv [n * n];
                u32 vtAAv [n * n];
                u32 vtAvbis [n * n];
                u32 vtAAvbis[n * n];
                u32 winv [n * n];
                u32 d[n];
                
                for (int i = 0; i < n * n; i++){
                        vtAv[i]  = 0;
                        vtAAv[i] = 0;
                }
                
                MPI_Scatterv( Av, counts, displacements, MPI_UINT32_T,  Avbis, line + line_reste, MPI_UINT32_T, 0, MPI_COMM_WORLD);
                MPI_Scatterv(  v, counts, displacements, MPI_UINT32_T,   vbis, line + line_reste, MPI_UINT32_T, 0, MPI_COMM_WORLD);
                MPI_Scatterv(tmp, counts, displacements, MPI_UINT32_T, tmpbis, line + line_reste, MPI_UINT32_T, 0, MPI_COMM_WORLD);
                
                block_dot_products(vtAvbis, vtAAvbis, (line + line_reste)/n, Avbis, vbis);
                
                MPI_Reduce(vtAvbis,   vtAv, n*n, MPI_UINT32_T, operation, 0, MPI_COMM_WORLD);
                MPI_Reduce(vtAAvbis, vtAAv, n*n, MPI_UINT32_T, operation, 0, MPI_COMM_WORLD);
                if ( rang == 0) {
                        stop = (semi_inverse(vtAv, winv, d) == 0);
                        
                        /* check that everything is working ; disable in production */
                        //correctness_tests(vtAv, vtAAv, winv, d);
                        
                        /*pour que les autres processus sachent quand s'arreter*/
                        MPI_Bcast(&stop, 1  , MPI_C_BOOL, 0,MPI_COMM_WORLD);
                        MPI_Bcast(vtAv , n*n, MPI_UINT32_T, 0,MPI_COMM_WORLD);
                        MPI_Bcast(vtAAv, n*n, MPI_UINT32_T, 0,MPI_COMM_WORLD);
                        MPI_Bcast(winv , n*n, MPI_UINT32_T, 0,MPI_COMM_WORLD);
                        MPI_Bcast(d    , n  , MPI_UINT32_T, 0,MPI_COMM_WORLD);
                       
                        if (stop){
                                break;
                        }
                } else {
                        MPI_Bcast(&stop, 1  , MPI_C_BOOL,   0,MPI_COMM_WORLD);
                        MPI_Bcast(vtAv,  n*n, MPI_UINT32_T, 0,MPI_COMM_WORLD);
                        MPI_Bcast(vtAAv, n*n, MPI_UINT32_T, 0,MPI_COMM_WORLD);
                        MPI_Bcast(winv , n*n, MPI_UINT32_T, 0,MPI_COMM_WORLD);
                        MPI_Bcast(d    , n  , MPI_UINT32_T, 0,MPI_COMM_WORLD);
                        
                        if (stop)
                                break;
                }
                orthogonalize(vbis, tmpbis, p, d, vtAv, vtAAv, winv, (line + line_reste)/n, Avbis);
                
                MPI_Gatherv(tmpbis, line+line_reste , MPI_UINT32_T, tmp,counts,displacements, MPI_UINT32_T, 0, MPI_COMM_WORLD);
                
                n_iterations++;
                 if ( rang == 0){
                        /* the next value of v is in tmp ; copy */
                        for (long i = 0; i < block_size; i++)
                                v[i] = tmp[i];

                        verbosity();
                }
        }
        printf("\n");

        if ( rang == 0){
                if (stop_after < 0) {
                        final_check(nrows, ncols, v, tmp);
                }
                printf("  - Terminated in %.1fs after %d iterations\n", wtime() - start, n_iterations);
        }
        
        free(tmp);
        free(Av);
        free(p);
        
        return v;
}


/**************************** dense vector block IO ************************/

void save_vector_block(char const * filename, int nrows, int ncols, u32 const * v)
{
        printf("Saving result in %s\n", filename);
        FILE * f = fopen(filename, "w");
        if (f == NULL)
                err(1, "cannot open %s", filename);
        fprintf(f, "%%%%MatrixMarket matrix array integer general\n");
        fprintf(f, "%%block of left-kernel vector computed by lanczos_modp\n");
        fprintf(f, "%d %d\n", nrows, ncols);
        for (long j = 0; j < ncols; j++)
                for (long i = 0; i < nrows; i++)
                        fprintf(f, "%d\n", v[i*n + j]);
        fclose(f);
}

/*************************** main function *********************************/
void share_matrix(struct sparsematrix_t * M){

        if ( rang == 0){
                int counts[proc];
                int displacements[proc];
                long nnz=M->nnz;
                long sub_nnz=nnz/proc;
                for(int i=0;i<proc;i++){
                        displacements[i]=i*sub_nnz;
                        if(i==proc-1){
                                counts[i]=sub_nnz+nnz%proc;
                                }
                        else{
                                counts[i]=sub_nnz;
                                }
                        }

                MPI_Bcast(&(nnz),1,MPI_LONG,0,MPI_COMM_WORLD);
                MPI_Bcast(&(M->nrows),1,MPI_INT,0,MPI_COMM_WORLD);
                MPI_Bcast(&(M->ncols),1,MPI_INT,0,MPI_COMM_WORLD);
                
                MPI_Scatterv(M->i,counts,displacements,MPI_INT,MPI_IN_PLACE,sub_nnz,MPI_INT,0,MPI_COMM_WORLD);
                MPI_Scatterv(M->j,counts,displacements,MPI_INT,MPI_IN_PLACE,sub_nnz,MPI_INT,0,MPI_COMM_WORLD);
                MPI_Scatterv(M->x,counts,displacements,MPI_UINT32_T,MPI_IN_PLACE,sub_nnz,MPI_INT,0,MPI_COMM_WORLD);
                M->nnz=sub_nnz;
        } else {

                int nrows = 0;
                int ncols = 0;
                long nnz  = 0;
                
                MPI_Bcast(&nnz,1,MPI_LONG,0,MPI_COMM_WORLD);
                MPI_Bcast(&nrows,1,MPI_INT,0,MPI_COMM_WORLD);
                MPI_Bcast(&ncols,1,MPI_INT,0,MPI_COMM_WORLD);
                if(rang==proc-1){
                        nnz=nnz/proc+nnz%proc;
                        }
                else{
                        nnz=nnz/proc;
                        }
                M->nrows = nrows;
                M->ncols = ncols;
                M->nnz   = nnz;
               
                int *Mi = malloc(nnz * sizeof(*Mi));
                int *Mj = malloc(nnz * sizeof(*Mj));
                u32 *Mx = malloc(nnz * sizeof(*Mx));
                
                if(Mi==NULL && Mj==NULL && Mx==NULL){
                        errx(1, "impossible d'allouer Matrice share matrix");
                        }
                MPI_Scatterv(NULL,NULL,NULL,MPI_INT,Mi,nnz,MPI_INT,0,MPI_COMM_WORLD);
                MPI_Scatterv(NULL,NULL,NULL,MPI_INT,Mj,nnz,MPI_INT,0,MPI_COMM_WORLD);
                MPI_Scatterv(NULL,NULL,NULL,MPI_INT,Mx,nnz,MPI_UINT32_T,0,MPI_COMM_WORLD);
                M->i = Mi;
                M->j = Mj;
                M->x = Mx;
        }

        }
int main(int argc, char ** argv)
{

        //MPI_Init(&argc, &argv);
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        //MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &proc);
        MPI_Comm_rank(MPI_COMM_WORLD, &rang);

        process_command_line_options(argc, argv);
                
        struct sparsematrix_t M;
        
        
        if(rang==0){
                sparsematrix_mm_load(&M,matrix_filename);
                }
        share_matrix(&M);
        
        u32 *kernel = block_lanczos(&M, n, right_kernel);
        
        if ( rang == 0 ){
                if (kernel_filename)
                        save_vector_block(kernel_filename, right_kernel ? M.ncols : M.nrows, n, kernel);
                else
                        printf("Not saving result (no --output given)\n");
        }
        free(kernel);
        
        MPI_Finalize();

        return 0;

}