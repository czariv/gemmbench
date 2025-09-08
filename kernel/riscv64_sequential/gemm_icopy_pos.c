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

  long i, j;

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

  j = n;
  if (j > 0){
    do{
	  aoffset1  = aoffset;
      aoffset2  = aoffset + lda;
      aoffset += 128;

      i = (m >> 7);
      if (i > 0){
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
		*(boffset + 32) = *(aoffset1 + 32);
		*(boffset + 33) = *(aoffset1 + 33);
		*(boffset + 34) = *(aoffset1 + 34);
		*(boffset + 35) = *(aoffset1 + 35);
		*(boffset + 36) = *(aoffset1 + 36);
		*(boffset + 37) = *(aoffset1 + 37);
		*(boffset + 38) = *(aoffset1 + 38);
		*(boffset + 39) = *(aoffset1 + 39);
		*(boffset + 40) = *(aoffset1 + 40);
		*(boffset + 41) = *(aoffset1 + 41);
		*(boffset + 42) = *(aoffset1 + 42);
		*(boffset + 43) = *(aoffset1 + 43);
		*(boffset + 44) = *(aoffset1 + 44);
		*(boffset + 45) = *(aoffset1 + 45);
		*(boffset + 46) = *(aoffset1 + 46);
		*(boffset + 47) = *(aoffset1 + 47);
		*(boffset + 48) = *(aoffset1 + 48);
		*(boffset + 49) = *(aoffset1 + 49);
		*(boffset + 50) = *(aoffset1 + 50);
		*(boffset + 51) = *(aoffset1 + 51);
		*(boffset + 52) = *(aoffset1 + 52);
		*(boffset + 53) = *(aoffset1 + 53);
		*(boffset + 54) = *(aoffset1 + 54);
		*(boffset + 55) = *(aoffset1 + 55);
		*(boffset + 56) = *(aoffset1 + 56);
		*(boffset + 57) = *(aoffset1 + 57);
		*(boffset + 58) = *(aoffset1 + 58);
		*(boffset + 59) = *(aoffset1 + 59);
		*(boffset + 60) = *(aoffset1 + 60);
		*(boffset + 61) = *(aoffset1 + 61);
		*(boffset + 62) = *(aoffset1 + 62);
		*(boffset + 63) = *(aoffset1 + 63);
		*(boffset + 64) = *(aoffset1 + 64);
		*(boffset + 65) = *(aoffset1 + 65);
		*(boffset + 66) = *(aoffset1 + 66);
		*(boffset + 67) = *(aoffset1 + 67);
		*(boffset + 68) = *(aoffset1 + 68);
		*(boffset + 69) = *(aoffset1 + 69);
		*(boffset + 70) = *(aoffset1 + 70);
		*(boffset + 71) = *(aoffset1 + 71);
		*(boffset + 72) = *(aoffset1 + 72);
		*(boffset + 73) = *(aoffset1 + 73);
		*(boffset + 74) = *(aoffset1 + 74);
		*(boffset + 75) = *(aoffset1 + 75);
		*(boffset + 76) = *(aoffset1 + 76);
		*(boffset + 77) = *(aoffset1 + 77);
		*(boffset + 78) = *(aoffset1 + 78);
		*(boffset + 79) = *(aoffset1 + 79);
		*(boffset + 80) = *(aoffset1 + 80);
		*(boffset + 81) = *(aoffset1 + 81);
		*(boffset + 82) = *(aoffset1 + 82);
		*(boffset + 83) = *(aoffset1 + 83);
		*(boffset + 84) = *(aoffset1 + 84);
		*(boffset + 85) = *(aoffset1 + 85);
		*(boffset + 86) = *(aoffset1 + 86);
		*(boffset + 87) = *(aoffset1 + 87);
		*(boffset + 88) = *(aoffset1 + 88);
		*(boffset + 89) = *(aoffset1 + 89);
		*(boffset + 90) = *(aoffset1 + 90);
		*(boffset + 91) = *(aoffset1 + 91);
		*(boffset + 92) = *(aoffset1 + 92);
		*(boffset + 93) = *(aoffset1 + 93);
		*(boffset + 94) = *(aoffset1 + 94);
		*(boffset + 95) = *(aoffset1 + 95);
		*(boffset + 96) = *(aoffset1 + 96);
		*(boffset + 97) = *(aoffset1 + 97);
		*(boffset + 98) = *(aoffset1 + 98);
		*(boffset + 99) = *(aoffset1 + 99);
		*(boffset +100) = *(aoffset1 +100);
		*(boffset +101) = *(aoffset1 +101);
		*(boffset +102) = *(aoffset1 +102);
		*(boffset +103) = *(aoffset1 +103);
		*(boffset +104) = *(aoffset1 +104);
		*(boffset +105) = *(aoffset1 +105);
		*(boffset +106) = *(aoffset1 +106);
		*(boffset +107) = *(aoffset1 +107);
		*(boffset +108) = *(aoffset1 +108);
		*(boffset +109) = *(aoffset1 +109);
		*(boffset +110) = *(aoffset1 +110);
		*(boffset +111) = *(aoffset1 +111);
		*(boffset +112) = *(aoffset1 +112);
		*(boffset +113) = *(aoffset1 +113);
		*(boffset +114) = *(aoffset1 +114);
		*(boffset +115) = *(aoffset1 +115);
		*(boffset +116) = *(aoffset1 +116);
		*(boffset +117) = *(aoffset1 +117);
		*(boffset +118) = *(aoffset1 +118);
		*(boffset +119) = *(aoffset1 +119);
		*(boffset +120) = *(aoffset1 +120);
		*(boffset +121) = *(aoffset1 +121);
		*(boffset +122) = *(aoffset1 +122);
		*(boffset +123) = *(aoffset1 +123);
		*(boffset +124) = *(aoffset1 +124);
		*(boffset +125) = *(aoffset1 +125);
		*(boffset +126) = *(aoffset1 +126);
		*(boffset +127) = *(aoffset1 +127);

		aoffset1 +=  128;
		boffset  += 128;
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
