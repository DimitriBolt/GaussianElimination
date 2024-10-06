/*----------------------------------------------------------------
* File:     main.c
*----------------------------------------------------------------
*
* Author:   Marek Rychlik (rychlik@arizona.edu)
* Date:     Sun Sep 22 10:54:06 2024
* Copying:  (C) Marek Rychlik, 2020. All rights reserved.
*
*----------------------------------------------------------------*/
/* Driver for gauss_solve.c */

#define _GNU_SOURCE
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <fenv.h>
#include <setjmp.h>

#include "gauss_solve.h"
#include "helpers.h"

/* Size of the matrix */


#define N  3

void test_gauss_solve()
{
  printf("Entering function: %s\n", __func__);
  
  const double A0[N][N] = {
    {2, 3, -1},
    {4, 1, 2},
    {-2, 7, 2}
  };

  const double b0[N] = {5, 6, 3};

  double A[N][N], b[N], x[N], y[N];

  /* Create copies of the matrices.
     NOTE: the copies will get destroyed. */
  memcpy(A, A0, sizeof(A0));
  memcpy(b, b0, sizeof(b0));  

  gauss_solve_in_place(N, A, b);

  memcpy(x, b, sizeof(b0));
  matrix_times_vector(N, A0, x, y);

  double eps = 1e-6, dist = norm_dist(N, b0, y);
  assert( dist < eps);

  /* Print x */
  puts("x:\n");
  print_vector(N, x);

  /* Print U */
  puts("U:\n");
  print_matrix(N, A, FLAG_UPPER_PART);
  
  /* Print L */
  puts("L:\n");
  print_matrix(N, A, FLAG_LOWER_PART);
}

jmp_buf env;  // Buffer to store the state for setjmp/longjmp

void test_gauss_solve_with_zero_pivot()
{
  printf("Entering function: %s\n", __func__);
  
  double A[N][N] = {
    {0, 3, -1},
    {4, 1, 2},
    {-2, 7, 2}
  };

  double b[N] = {5, 6, 3};

  // Save the program state with setjmp
  if (setjmp(env) == 0) {
    gauss_solve_in_place(N, A, b);
    print_matrix(N, A, FLAG_LOWER_PART);
  } else {
    // This block is executed when longjmp is called
    printf("Returned to main program flow after exception\n");
  }
  
}

void test_lu_in_place()
{
  printf("Entering function: %s\n", __func__);

  const double A0[N][N] = {
    {2, 3, -1},
    {4, 1, 2},
    {-2, 7, 2}
  };

  const double b0[N] = {5, 6, 3};

  double A[N][N];

  memcpy(A, A0, sizeof(A0));

  lu_in_place(N, A);


  /* Print U */
  puts("U:\n");
  print_matrix(N, A, FLAG_UPPER_PART);
  
  /* Print L */
  puts("L:\n");
  print_matrix(N, A, FLAG_LOWER_PART);

  lu_in_place_reconstruct(N, A);

  /* Print U */
  puts("Reconstructed A:\n");
  print_matrix(N, A, FLAG_WHOLE);

  memcpy(A, A0, sizeof(A0));
  puts("Original A:\n");
  print_matrix(N, A, FLAG_WHOLE);

  double eps = 1e-6;
  assert(frobenius_norm_dist(N, A, A0) < eps);
}

void benchmark_test(int n)
{
  /* Allocate matrix on stack */
  double A0[n][n], A[n][n];
  generate_random_matrix(n, A0);
  copy_matrix(n, A0, A);

  lu_in_place(N, A);
  lu_in_place_reconstruct(N, A);

  double eps = 1e-6;
  assert(frobenius_norm_dist(N, A0, A) < eps);

}

void benchmark_test_dynamic(int n)
{
  /* Allocate memory */
  double *store = (double *)malloc(n * n * sizeof(double));
  double (*A)[n] = (double (*)[n])store;
  assert(A);

  double *store_copy = (double *)malloc(n * n * sizeof(double));
  double (*A_copy)[n] = (double (*)[n])store_copy;
  assert(A_copy);

  generate_random_matrix(n, A);
  copy_matrix(n, A, A_copy);

  puts("Random matrix A:\n");
  print_matrix(n, A, FLAG_WHOLE);

  lu_in_place(n, A);
  lu_in_place_reconstruct(n, A);

  double eps = 1e-6;
  assert(frobenius_norm_dist(n, A_copy, A) < eps);

  /* Ensure memory is deallocated */
  free(store);
  free(store_copy);
}

void benchmark_test_dynamic_alt(int n)
{
  /* Allocate memory */
  double (*A)[n] = NULL, (*A_copy)[n] = NULL;
  create_matrix(n, &A);
  assert(A);
  create_matrix(n, &A_copy);  
  assert(A_copy);

  generate_random_matrix(n, A);
  copy_matrix(n, A, A_copy);

  lu_in_place(n, A);

  lu_in_place_reconstruct(n, A);

  double eps = 1e-4, dist = frobenius_norm_dist(n, A_copy, A);
  assert(dist < eps);


  /* Ensure memory is deallocated */
  destroy_matrix(n, A);
  destroy_matrix(n, A_copy);
}

void fpe_handler(int sig) {
  printf("Entering %s...\n", __func__);
  if(sig == SIGFPE) {
    printf("Floating point exception occurred, ignoring...\n");
    longjmp(env, 1);  // Jump back to where setjmp was called
  }
}

int main()
{
  // Enable trapping for specific floating-point exceptions
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  sighandler_t old_handler = signal(SIGFPE, fpe_handler);

  // test_gauss_solve();
  // test_lu_in_place();
  // benchmark_test(5);
  // benchmark_test_dynamic(5);
  // benchmark_test_dynamic_alt(2000);
  // test_gauss_solve_with_zero_pivot();  
  exit(EXIT_SUCCESS);
}
