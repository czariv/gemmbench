/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#include "interface.h"

int gemm_icopy_pos(long m, long n, float *a, long lda, float *b){

  long i, j, h;

  float *aoffset;
  float *aoffset1, *aoffset2;
  float *boffset;

  float ctemp01, ctemp02, ctemp03, ctemp04;
  float ctemp05, ctemp06, ctemp07, ctemp08;
  float ctemp09, ctemp10, ctemp11, ctemp12;
  float ctemp13, ctemp14, ctemp15, ctemp16;

  aoffset   = a;
  boffset   = b;

#if 0
  fprintf(stderr, "m = %d n = %d\n", m, n);
#endif
  h = (m >> 4);
  if (h>0){
   do{
    aoffset = a + lda * ((m >> 4) - h);
  	j = (n >> 3);
	if (j > 0){
		do{
		aoffset1  = aoffset;
		aoffset2  = aoffset + lda;
		aoffset += 1024;

		i = (m >> 5);
		if (i > 0){
		do{
		*(boffset +  0) = *(aoffset1 +  0);
		*(boffset +  1) = *(aoffset1 +  1);
		*(boffset +  2) = *(aoffset1 +  2);
		*(boffset +  3) = *(aoffset1 +  3);
		*(boffset +  4) = *(aoffset1 +  4);
		*(boffset +  5) = *(aoffset1 +  5);
		*(boffset +  6) = *(aoffset1 +  6);
		*(boffset +  7) = *(aoffset1 +  7);

		*(boffset +  8) = *(aoffset1 +  8);
		*(boffset +  9) = *(aoffset1 +  9);
		*(boffset + 10) = *(aoffset1 + 10);
		*(boffset + 11) = *(aoffset1 + 11);
		*(boffset + 12) = *(aoffset1 + 12);
		*(boffset + 13) = *(aoffset1 + 13);
		*(boffset + 14) = *(aoffset1 + 14);
		*(boffset + 15) = *(aoffset1 + 15);

		*(boffset + 16) = *(aoffset1 + 16);
		*(boffset + 17) = *(aoffset1 + 17);
		*(boffset + 18) = *(aoffset1 + 18);
		*(boffset + 19) = *(aoffset1 + 19);
		*(boffset + 20) = *(aoffset1 + 20);
		*(boffset + 21) = *(aoffset1 + 21);
		*(boffset + 22) = *(aoffset1 + 22);
		*(boffset + 23) = *(aoffset1 + 23);

		*(boffset + 24) = *(aoffset1 + 24);
		*(boffset + 25) = *(aoffset1 + 25);
		*(boffset + 26) = *(aoffset1 + 26);
		*(boffset + 27) = *(aoffset1 + 27);
		*(boffset + 28) = *(aoffset1 + 28);
		*(boffset + 29) = *(aoffset1 + 29);
		*(boffset + 30) = *(aoffset1 + 30);
		*(boffset + 31) = *(aoffset1 + 31);

		aoffset1 +=  32;
		boffset  += 32;

		i --;
		}while(i > 0);
		}

		if (m & 1){
		ctemp01 = *(aoffset1 +  0);
		ctemp02 = *(aoffset1 +  1);
		ctemp03 = *(aoffset1 +  2);
		ctemp04 = *(aoffset1 +  3);
		ctemp05 = *(aoffset1 +  4);
		ctemp06 = *(aoffset1 +  5);
		ctemp07 = *(aoffset1 +  6);
		ctemp08 = *(aoffset1 +  7);
		ctemp09 = *(aoffset1 +  8);
		ctemp10 = *(aoffset1 +  9);
		ctemp11 = *(aoffset1 + 10);
		ctemp12 = *(aoffset1 + 11);
		ctemp13 = *(aoffset1 + 12);
		ctemp14 = *(aoffset1 + 13);
		ctemp15 = *(aoffset1 + 14);
		ctemp16 = *(aoffset1 + 15);

		*(boffset +  0) = ctemp01;
		*(boffset +  1) = ctemp02;
		*(boffset +  2) = ctemp03;
		*(boffset +  3) = ctemp04;
		*(boffset +  4) = ctemp05;
		*(boffset +  5) = ctemp06;
		*(boffset +  6) = ctemp07;
		*(boffset +  7) = ctemp08;

		*(boffset +  8) = ctemp09;
		*(boffset +  9) = ctemp10;
		*(boffset + 10) = ctemp11;
		*(boffset + 11) = ctemp12;
		*(boffset + 12) = ctemp13;
		*(boffset + 13) = ctemp14;
		*(boffset + 14) = ctemp15;
		*(boffset + 15) = ctemp16;

		boffset   += 16;
		}

		j--;
		}while(j > 0);
	} /* end of if(j > 0) */
   h--;
   }while(h > 0);
  }

  if (n & 8){
    aoffset1  = aoffset;
    aoffset2  = aoffset + lda;
    aoffset += 8;

    i = (m >> 1);
    if (i > 0){
      do{
	ctemp01 = *(aoffset1 +  0);
	ctemp02 = *(aoffset1 +  1);
	ctemp03 = *(aoffset1 +  2);
	ctemp04 = *(aoffset1 +  3);
	ctemp05 = *(aoffset1 +  4);
	ctemp06 = *(aoffset1 +  5);
	ctemp07 = *(aoffset1 +  6);
	ctemp08 = *(aoffset1 +  7);

	ctemp09 = *(aoffset2 +  0);
	ctemp10 = *(aoffset2 +  1);
	ctemp11 = *(aoffset2 +  2);
	ctemp12 = *(aoffset2 +  3);
	ctemp13 = *(aoffset2 +  4);
	ctemp14 = *(aoffset2 +  5);
	ctemp15 = *(aoffset2 +  6);
	ctemp16 = *(aoffset2 +  7);

	*(boffset +  0) = ctemp01;
	*(boffset +  1) = ctemp02;
	*(boffset +  2) = ctemp03;
	*(boffset +  3) = ctemp04;
	*(boffset +  4) = ctemp05;
	*(boffset +  5) = ctemp06;
	*(boffset +  6) = ctemp07;
	*(boffset +  7) = ctemp08;

	*(boffset +  8) = ctemp09;
	*(boffset +  9) = ctemp10;
	*(boffset + 10) = ctemp11;
	*(boffset + 11) = ctemp12;
	*(boffset + 12) = ctemp13;
	*(boffset + 13) = ctemp14;
	*(boffset + 14) = ctemp15;
	*(boffset + 15) = ctemp16;

	aoffset1 +=  2 * lda;
	aoffset2 +=  2 * lda;
	boffset   += 16;

	i --;
      }while(i > 0);
    }

    if (m & 1){
      ctemp01 = *(aoffset1 +  0);
      ctemp02 = *(aoffset1 +  1);
      ctemp03 = *(aoffset1 +  2);
      ctemp04 = *(aoffset1 +  3);
      ctemp05 = *(aoffset1 +  4);
      ctemp06 = *(aoffset1 +  5);
      ctemp07 = *(aoffset1 +  6);
      ctemp08 = *(aoffset1 +  7);

      *(boffset +  0) = ctemp01;
      *(boffset +  1) = ctemp02;
      *(boffset +  2) = ctemp03;
      *(boffset +  3) = ctemp04;
      *(boffset +  4) = ctemp05;
      *(boffset +  5) = ctemp06;
      *(boffset +  6) = ctemp07;
      *(boffset +  7) = ctemp08;

      boffset   += 8;
    }
  }

  if (n & 4){
    aoffset1  = aoffset;
    aoffset2  = aoffset + lda;
    aoffset += 4;

    i = (m >> 1);
    if (i > 0){
      do{
	ctemp01 = *(aoffset1 +  0);
	ctemp02 = *(aoffset1 +  1);
	ctemp03 = *(aoffset1 +  2);
	ctemp04 = *(aoffset1 +  3);

	ctemp05 = *(aoffset2 +  0);
	ctemp06 = *(aoffset2 +  1);
	ctemp07 = *(aoffset2 +  2);
	ctemp08 = *(aoffset2 +  3);

	*(boffset +  0) = ctemp01;
	*(boffset +  1) = ctemp02;
	*(boffset +  2) = ctemp03;
	*(boffset +  3) = ctemp04;
	*(boffset +  4) = ctemp05;
	*(boffset +  5) = ctemp06;
	*(boffset +  6) = ctemp07;
	*(boffset +  7) = ctemp08;

	aoffset1 +=  2 * lda;
	aoffset2 +=  2 * lda;
	boffset   += 8;

	i --;
      }while(i > 0);
    }

    if (m & 1){
      ctemp01 = *(aoffset1 +  0);
      ctemp02 = *(aoffset1 +  1);
      ctemp03 = *(aoffset1 +  2);
      ctemp04 = *(aoffset1 +  3);

      *(boffset +  0) = ctemp01;
      *(boffset +  1) = ctemp02;
      *(boffset +  2) = ctemp03;
      *(boffset +  3) = ctemp04;

      boffset   += 4;
    }
  }

  if (n & 2){
    aoffset1  = aoffset;
    aoffset2  = aoffset + lda;
    aoffset += 2;

    i = (m >> 1);
    if (i > 0){
      do{
	ctemp01 = *(aoffset1 +  0);
	ctemp02 = *(aoffset1 +  1);
	ctemp03 = *(aoffset2 +  0);
	ctemp04 = *(aoffset2 +  1);

	*(boffset +  0) = ctemp01;
	*(boffset +  1) = ctemp02;
	*(boffset +  2) = ctemp03;
	*(boffset +  3) = ctemp04;

	aoffset1 +=  2 * lda;
	aoffset2 +=  2 * lda;
	boffset   += 4;

	i --;
      }while(i > 0);
    }

    if (m & 1){
      ctemp01 = *(aoffset1 +  0);
      ctemp02 = *(aoffset1 +  1);

      *(boffset +  0) = ctemp01;
      *(boffset +  1) = ctemp02;
      boffset   += 2;
    }
  }

  if (n & 1){
    aoffset1  = aoffset;
    aoffset2  = aoffset + lda;

    i = (m >> 1);
    if (i > 0){
      do{
	ctemp01 = *(aoffset1 +  0);
	ctemp02 = *(aoffset2 +  0);

	*(boffset +  0) = ctemp01;
	*(boffset +  1) = ctemp02;

	aoffset1 +=  2 * lda;
	aoffset2 +=  2 * lda;
	boffset   += 2;

	i --;
      }while(i > 0);
    }

    if (m & 1){
      ctemp01 = *(aoffset1 +  0);
      *(boffset +  0) = ctemp01;
      // boffset   += 1;
    }
  }

  return 0;
}
