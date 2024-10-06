/*----------------------------------------------------------------
* File:     gauss_solve.h
*----------------------------------------------------------------
*
* Author:   Marek Rychlik (rychlik@arizona.edu)
* Date:     Sun Sep 22 15:40:51 2024
* Copying:  (C) Marek Rychlik, 2020. All rights reserved.
*
*----------------------------------------------------------------*/

#ifndef GAUSS_SOLVE_H
#define GAUSS_SOLVE_H


/* An idiomatic way to swap two l-values X and Y of type TYPE in C
   Example:
   int x = 1; int y = 2; SWAP(x, y, int);
   Now x==2 and y==1.
*/
#define SWAP(X, Y, TYPE) do {			\
    TYPE tmp = (X);				\
    (X) = (Y);					\
    (Y) = tmp;					\
  } while(0)


void gauss_solve_in_place(const int n, double A[n][n], double b[n]);
void lu_in_place(const int n, double A[n][n]);
void lu_in_place_reconstruct(int n, double A[n][n]);
void plu(int n, double A[n][n], int P[n]);

#endif
