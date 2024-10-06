/*----------------------------------------------------------------
* File:     gauss_solve.c
*----------------------------------------------------------------
*
* Author:   Marek Rychlik (rychlik@arizona.edu)
* Date:     Sun Sep 22 15:40:29 2024
* Copying:  (C) Marek Rychlik, 2020. All rights reserved.
*
*----------------------------------------------------------------*/
#include "gauss_solve.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void gauss_solve_in_place(const int n, double A[n][n], double b[n])
{
  for(int k = 0; k < n; ++k) {
    for(int i = k+1; i < n; ++i) {
      /* Store the multiplier into A[i][k] as it would become 0 and be
	 useless */
      A[i][k] /= A[k][k];
      for( int j = k+1; j < n; ++j) {
	A[i][j] -= A[i][k] * A[k][j];
      }
      b[i] -= A[i][k] * b[k];
    }
  } /* End of Gaussian elimination, start back-substitution. */
  for(int i = n-1; i >= 0; --i) {
    for(int j = i+1; j<n; ++j) {
      b[i] -= A[i][j] * b[j];
    }
    b[i] /= A[i][i];
  } /* End of back-substitution. */
}

void lu_in_place(const int n, double A[n][n])
{
  for(int k = 0; k < n; ++k) {
    for(int i = k; i < n; ++i) {
      for(int j=0; j<k; ++j) {
	/* U[k][i] -= L[k][j] * U[j][i] */
	A[k][i] -=  A[k][j] * A[j][i]; 
      }
    }
    for(int i = k+1; i<n; ++i) {
      for(int j=0; j<k; ++j) {
	/* L[i][k] -= A[i][k] * U[j][k] */
	A[i][k] -= A[i][j]*A[j][k]; 
      }
      /* L[k][k] /= U[k][k] */
      A[i][k] /= A[k][k];	
    }
  }
}

void lu_in_place_reconstruct(int n, double A[n][n])
{
  for(int k = n-1; k >= 0; --k) {
    for(int i = k+1; i<n; ++i) {
      A[i][k] *= A[k][k];
      for(int j=0; j<k; ++j) {
	A[i][k] += A[i][j]*A[j][k];
      }
    }
    for(int i = k; i < n; ++i) {
      for(int j=0; j<k; ++j) {
	A[k][i] +=  A[k][j] * A[j][i];
      }
    }
  }
}


void plu(int n, double A[n][n], int P[n]) {
    int i, j, k, maxIndex;
    double maxVal, temp;
    double L[n][n], U[n][n];

    // Initialize permutation matrix P to the identity permutation
    for (i = 0; i < n; i++) {
        P[i] = i;
    }

    // Initialize L to zero and copy A into U
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            L[i][j] = 0.0;
            U[i][j] = A[i][j];
        }
    }

    // Perform PA=LU decomposition with partial pivoting
    for (k = 0; k < n - 1; k++) {
        // Find the pivot row
        maxIndex = k;
        maxVal = fabs(U[k][k]);
        for (i = k + 1; i < n; i++) {
            if (fabs(U[i][k]) > maxVal) {
                maxVal = fabs(U[i][k]);
                maxIndex = i;
            }
        }

        // Swap rows in U and adjust the permutation matrix P
        if (maxIndex != k) {
            for (j = 0; j < n; j++) {
                temp = U[k][j];
                U[k][j] = U[maxIndex][j];
                U[maxIndex][j] = temp;
            }
            int tempP = P[k];
            P[k] = P[maxIndex];
            P[maxIndex] = tempP;

            // Swap rows in L for columns < k
            for (j = 0; j < k; j++) {
                temp = L[k][j];
                L[k][j] = L[maxIndex][j];
                L[maxIndex][j] = temp;
            }
        }

        // Compute the multipliers and eliminate entries below the pivot
        for (i = k + 1; i < n; i++) {
            L[i][k] = U[i][k] / U[k][k];
            for (j = k; j < n; j++) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }
    }
    for (i = 0; i<n; i++){
      for (j=0; j<n; j++){
        if (j>=i){
          A[i][j] = U[i][j];
        } else {
          A[i][j] = L[i][j];
          }
      }
    }


    // Set the diagonal elements of L to 1
    for (i = 0; i < n; i++) {
        L[i][i] = 1.0;
    }

    // Output matrices P, L, and U for verification (optional)
    printf("Permutation vector P:\n");
    for (i = 0; i < n; i++) {
        printf("%d ", P[i]);
    }
    printf("\n");

    printf("Matrix L:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%0.2f ", L[i][j]);
        }
        printf("\n");
    }

    printf("Matrix U:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%0.2f ", U[i][j]);
        }
        printf("\n");
    }
}
