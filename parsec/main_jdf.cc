/* Copyright 2018 Los Alamos National Laboratory
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdarg.h>

#include "core.h"
#include "common.h"
#include <parsec/execution_stream.h>
//#include <dplasmatypes.h>
#include <data_dist/matrix/two_dim_rectangle_cyclic.h>
#include <parsec/arena.h>

/* timings */
#if defined( PARSEC_HAVE_MPI)
#define MPI_TIMING
#endif
#include "timer.h"
#include "benchmark_internal.h"

#define MAX_ARGS  4
#define APPLY  0

#define VERBOSE_LEVEL 0

#define USE_CORE_VERIFICATION
//#define TRACK_NB_TASKS

#if defined (TRACK_NB_TASKS)  
int nb_tasks_per_node[32];
#endif

char **extra_local_memory;

/* For timming */
double *timecount;
double *timecount_all;

/* Global */
int M, N, MB, NB, P, SMB, SNB, cores, nodes;

typedef struct matrix_s{
  two_dim_block_cyclic_t dcC;
  int M;
  int N;
  int K;
  int NRHS;
  int IB;
  int MB;
  int NB;
  int SMB;
  int SNB;
  int HMB;
  int HNB;
  int MT;
  int NT;
  int KT;
}matrix_t;

struct ParsecApp : public App {
  ParsecApp(int argc, char **argv);
  ~ParsecApp();
  void execute_main_loop();
  void execute_timestep(size_t idx, long t);
  void debug_printf(int verbose_level, const char *format, ...);
private:
  parsec_context_t* parsec;
  int rank;
  int nodes;
  int cores;
  int gpus;
  int P;
  int Q;
  matrix_t mat_array[10];
  int check;
  int loud;
  int scheduler;
  int iparam[IPARAM_SIZEOF];
  int nb_tasks;
  int nb_fields;
};

static int stencil_1d_init_ops(parsec_execution_stream_t *es,
                        const parsec_tiled_matrix_dc_t *descA,
                        void *_A, enum matrix_uplo uplo,
                        int m, int n, void *args)
{
    float *A = (float *)_A;
    int R = ((int *)args)[0];

    //printf("in data init (%d, %d): th_id: %d, core_id: %d, socket_id: %d; vp_p: %d, vp_q: %d\n", m, n, es->th_id, es->core_id, es->socket_id, ((two_dim_block_cyclic_t *)descA)->grid.vp_p, ((two_dim_block_cyclic_t *)descA)->grid.vp_q);

    for(int j = R; j < descA->nb - R; j++)
        for(int i = 0; i < descA->mb; i++)
            A[j*descA->mb+i] = (float)0.0;

    (void)es; (void)uplo; (void)m; (void)n;
    return 0;
}

ParsecApp::ParsecApp(int argc, char **argv)
  : App(argc, argv)
{ 
  int i, rank, ch;;

    /* Default */
    M = 8;
    N = 8;
    MB = 2;
    NB = 2;
    P = 1;
    SMB = 1;
    SNB = 1;
    cores = -1;
  
  nb_fields = 0;
  
  int nb_fields_arg = 0;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    nodes = 1;
    rank = 0;
#endif
  
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-field")) {
      nb_fields_arg = atol(argv[++i]);
    }

    if (!strcmp(argv[i], "-M")) {
      M = atol(argv[++i]);
    }

    if (!strcmp(argv[i], "-N")) {
      N = atol(argv[++i]);
    }

    if (!strcmp(argv[i], "-t")) {
      MB = atol(argv[++i]);
    }

    if (!strcmp(argv[i], "-T")) {
      NB = atol(argv[++i]);
    }

    if (!strcmp(argv[i], "-s")) {
      SMB = atol(argv[++i]);
    }

    if (!strcmp(argv[i], "-S")) {
      SNB = atol(argv[++i]);
    }

    if (!strcmp(argv[i], "-P")) {
      P = atol(argv[++i]);
    }

    if (!strcmp(argv[i], "-c")) {
      cores = atol(argv[++i]);
    }

    if (!strcmp(argv[i], "-help")) {
                fprintf(stderr,
                        "-M : row dimension (M) of the matrices (default: 4)\n"
                        "-N : column dimension (N) of the matrices (default: 4)\n"
                        "-t : row dimension (MB) of the tiles (default: 2)\n"
                        "-T : column dimension (NB) of the tiles (default: 2)\n"
                        "-s : rows of tiles in a supertile (default: 1)\n"
                        "-S : columns of tiles in a supertile (default: 1)\n"
                        "-P : rows (P) in the PxQ process grid (default: 1)\n"
                        "-c : number of cores used (default: -1)\n"
                        "\n");
    }
  }

    /* Once we got out arguments, we should pass whatever is left down */
    int parsec_argc, idx;
    char** parsec_argv = (char**)calloc(argc, sizeof(char*));
    parsec_argv[0] = argv[0];  /* the app name */
    for( idx = parsec_argc = 1;
         (idx < argc) && (0 != strcmp(argv[idx], "--")); idx++);
    if( idx != argc ) {
        for( parsec_argc = 1, idx++; idx < argc;
             parsec_argv[parsec_argc] = argv[idx], parsec_argc++, idx++);
    }

    /* Init PaRSEC */
    parsec = parsec_init(cores, &parsec_argc, &parsec_argv);
    free(parsec_argv);

    if( NULL == parsec ) {
        /* Failed to correctly initialize. In a correct scenario report
 *          * upstream, but in this particular case bail out.
 *                   */
        exit(-1);
    }

    /* If the number of cores has not been defined as a parameter earlier
 *      * update it with the default parameter computed in parsec_init. */
    if(cores <= 0)
    {
        int p, nb_total_comp_threads = 0;
        for(p = 0; p < parsec->nb_vp; p++) {
            nb_total_comp_threads += parsec->virtual_processes[p]->nb_cores;
        }
        cores = nb_total_comp_threads;
    }
  
#if defined (TRACK_NB_TASKS)    
  for (i = 0; i < cores; i++) {
      nb_tasks_per_node[i] = 0;
  }
#endif
  
  debug_printf(0, "init parsec, pid %d\n", getpid());
  
  size_t max_scratch_bytes_per_task = 0;
  
  int MB_cal = 0;
  
  for (i = 0; i < graphs.size(); i++) {
    TaskGraph &graph = graphs[i];
    matrix_t &mat = mat_array[i];
    
    if (nb_fields_arg > 0) {
      nb_fields = nb_fields_arg;
    } else {
      nb_fields = graph.timesteps;
    }
    
    MB_cal = sqrt(graph.output_bytes_per_task / sizeof(float));
    
    if (MB_cal > iparam[IPARAM_MB]) {
      MB = MB_cal;
      NB = NB;
    }
    
    N = graph.max_width * MB;
    M = nb_fields * MB;
  
    /* Set P to 1 */
    if( P != 1 ) {
       P = 1;
       Q = nodes;
       if( 0 == rank )
           printf("Warnning: set P = 1 Q = %d\n", nodes);
    }

    int nodes_vp = 2;
    /* Set P to 1 */
    if( (N/NB % nodes_vp == 0 && SNB != N/NB/nodes_vp)
        || (N/NB % nodes_vp != 0 && SNB != N/NB/nodes_vp + 1) ) {
        if( 0 == N/NB % nodes_vp )
            SNB = N/NB/nodes_vp;
        else
            SNB = N/NB/nodes_vp + 1;

        if( 0 == rank )
           printf("WARNNING: set distribution to two dim block; SNB = %d\n", SNB);
    }

    {
        fprintf(stderr, "#+++++ nodes x cores        : %d x %d\n", nodes, cores);
        fprintf(stderr, "#+++++ P x Q                : %d x %d\n", P, nodes/P);
        fprintf(stderr, "#+++++ M x N                : %d x %d\n", M, N);
        fprintf(stderr, "#+++++ MB x NB              : %d x %d\n", MB, NB);
        fprintf(stderr, "#+++++ SMB x SNB            : %d x %d\n", SMB, SNB); 
    }

    mat.M     = M;
    mat.N     = N;
    mat.MB    = MB;
    mat.NB    = NB;
    mat.SMB   = SMB;
    mat.SNB   = SNB;
    mat.MT    = (mat.M%mat.MB==0) ? (mat.M/mat.MB) : (mat.M/mat.MB+1);
    mat.NT    = (mat.N%mat.NB==0) ? (mat.N/mat.NB) : (mat.N/mat.NB+1);

    debug_printf(0, "output_bytes_per_task %d, mb %d, nb %d\n", graph.output_bytes_per_task, mat.MB, mat.NB);
    assert(graph.output_bytes_per_task <= sizeof(float) * mat.MB * mat.NB);
  
    two_dim_block_cyclic_init(&mat.dcC, matrix_RealFloat, matrix_Tile,
                               nodes, rank, mat.MB, mat.NB, mat.M, mat.N, 0, 0,
                               mat.M, mat.N, mat.SMB, mat.SNB, P);

    parsec_data_collection_set_key((parsec_data_collection_t*)&(mat.dcC), "dcC"); 

#if APPLY
    mat.dcC.mat = parsec_data_allocate((size_t)mat.dcC.super.nb_local_tiles * \
                                   (size_t)mat.dcC.super.bsiz *      \
                                   (size_t)parsec_datadist_getsizeoftype(mat.dcC.super.mtype)); \
    /* Init 0.0
     */
    int *op_args = (int *)malloc(sizeof(int));
    *op_args = 0;
    parsec_apply( parsec, matrix_UpperLower,
                  (parsec_tiled_matrix_dc_t *)&mat.dcC,
                  (tiled_matrix_unary_op_t)stencil_1d_init_ops, op_args);
#else
    parsec_benchmark_data_init(parsec, (parsec_tiled_matrix_dc_t *)&mat.dcC);
#endif

    /* For timming */
    timecount = (double *)calloc(cores, sizeof(double));
    if( 0 == rank )
        timecount_all = (double *)calloc(nodes*cores, sizeof(double));

    /* matrix generation */
    //dplasma_dplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC, Cseed);
                            
    if (graph.scratch_bytes_per_task > max_scratch_bytes_per_task) {
      max_scratch_bytes_per_task = graph.scratch_bytes_per_task;
    }
    
  }
  
  nb_tasks = 0;
  
  extra_local_memory = (char**)malloc(sizeof(char*) * cores);
  assert(extra_local_memory != NULL);
  for (i = 0; i < cores; i++) {
    if (max_scratch_bytes_per_task > 0) {
      extra_local_memory[i] = (char*)malloc(sizeof(char)*max_scratch_bytes_per_task);
    } else {
      extra_local_memory[i] = NULL;
    }
  }
  
  debug_printf(0, "max_scratch_bytes_per_task %lld\n", max_scratch_bytes_per_task);
}

ParsecApp::~ParsecApp()
{
  int i; 
  
  debug_printf(0, "clean up parsec\n");
  
  for (i = 0; i < cores; i++) {
    if (extra_local_memory[i] != NULL) {
      free(extra_local_memory[i]);
      extra_local_memory[i] = NULL;
    }
  }
  free(extra_local_memory);
  extra_local_memory = NULL;
  
}

void ParsecApp::execute_main_loop()
{
  
  if (rank == 0) {
    display();
  }
  
  //sleep(10);
  
  /* #### parsec context Starting #### */
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    Timer::time_start();
  }

  int x, y;
  
  parsec_taskpool_t* tp[10];
  for (int i = 0; i < graphs.size(); i++) {
    const TaskGraph &g = graphs[i];
    matrix_t &mat = mat_array[i];

    debug_printf(0, "rank %d, pid %d, M %d, N %d, MT %d, NT %d, nb_fields %d, timesteps %d\n", rank, getpid(), mat.M, mat.N, mat.MT, mat.NT, nb_fields, g.timesteps);
    
    if (g.dependence == DependenceType::STENCIL_1D) {
      //parsec_stencil_1d(parsec, (parsec_tiled_matrix_dc_t *)&mat, g, nb_fields, g.timesteps, i, extra_local_memory);
      tp[i] = parsec_stencil_1d_New((parsec_tiled_matrix_dc_t *)&mat, g, nb_fields, g.timesteps, i, extra_local_memory); 
    } else if (g.dependence == DependenceType::NEAREST && g.radix == 5) {
      //parsec_nearest_radix_5(parsec, (parsec_tiled_matrix_dc_t *)&mat, g, nb_fields, g.timesteps, i, extra_local_memory);
      tp[i] = parsec_nearest_radix_5_New((parsec_tiled_matrix_dc_t *)&mat, g, nb_fields, g.timesteps, i, extra_local_memory); 
    } else if (g.dependence == DependenceType::SPREAD && g.radix == 5) {
      //parsec_spread_radix5_period3(parsec, (parsec_tiled_matrix_dc_t *)&mat, g, nb_fields, g.timesteps, i, extra_local_memory);
      tp[i] = parsec_spread_radix5_period3_New((parsec_tiled_matrix_dc_t *)&mat, g, nb_fields, g.timesteps, i, extra_local_memory); 
    } else {
      assert(0);
      parsec_benchmark(parsec, (parsec_tiled_matrix_dc_t *)&mat, g, nb_fields, g.timesteps, i, extra_local_memory);
    }
    assert(tp[i] != NULL);
    parsec_enqueue(parsec, tp[i]);
  }
  
  parsec_context_start(parsec);
  parsec_context_wait(parsec);
  
  for (int i = 0; i < graphs.size(); i++) {
    const TaskGraph &g = graphs[i];
    
    if (g.dependence == DependenceType::STENCIL_1D) {
      parsec_stencil_1d_Destruct(tp[i]);
    } else if (g.dependence == DependenceType::NEAREST && g.radix == 5) {
      parsec_nearest_radix_5_Destruct(tp[i]);
    } else if (g.dependence == DependenceType::SPREAD && g.radix == 5) {
      parsec_spread_radix5_period3_Destruct(tp[i]);
    } else {
      assert(0);
    }
    tp[i] = NULL;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  double elapsed;
  if (rank == 0) {
    elapsed = Timer::time_end();
    report_timing(elapsed);
    debug_printf(0, "[****] TIME(s) %12.5f : \tnb_tasks %d", elapsed, nb_tasks);
  }

#if defined (TRACK_NB_TASKS)    
  for (int i = 1; i < cores; i++) {
    nb_tasks_per_node[0] += nb_tasks_per_node[i];
  }
  printf("rank %d, nb_tasks %d\n", rank, nb_tasks_per_node[0]);
#endif

  /* For timming */
  MPI_Gather(timecount, cores, MPI_DOUBLE, timecount_all, cores, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if( 0 == rank ) {
    double time_sum = 0.0;
    double time_max = 0.0;
    for(int i = 0; i < cores * nodes; i++) {
        if( timecount_all[i] >= time_max )
            time_max = timecount_all[i];
        time_sum += timecount_all[i];
    }
    printf("\tKernel_time_max: %lf Kernel_time_avg: %lf Time_diff: %lf Time_sum: %lf\n", time_max, time_sum/cores, elapsed-time_sum/cores, time_sum);

    for(int i = 0; i < cores * nodes; i++) {
        printf(" %d time_for_cores: %lf\n", i, timecount_all[i]);
    }

  }

    /* Clean up parsec*/
    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

}

void ParsecApp::debug_printf(int verbose_level, const char *format, ...)
{
  if (verbose_level > VERBOSE_LEVEL) {
    return;
  }
  if (rank == 0) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
  }
}

int main(int argc, char ** argv)
{
  //printf("pid %d\n", getpid());
  //sleep(10);
  ParsecApp app(argc, argv);
  app.execute_main_loop();

  return 0;
}
