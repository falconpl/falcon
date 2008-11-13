/*
   The Falcon Programming Language
   FILE: test_dynlib.c

   Direct dynamic library interface for Falcon

   C library exporting some test functions that can be called
   and prototyped from Falcon.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 13 Nov 2008 19:25:55 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

   See the LICENSE file distributed with this package for licensing details.
*/

#include <falcon/setup.h>
#include <stdio.h>

//=====================================================
// Basic int/float calls
//

int FALCON_DYN_SYM call_p0_ri()
{
   return 100;
}

int FALCON_DYN_SYM call_p3i_ri( int a, int b, int c )
{
   return a + b + c;
}

int FALCON_DYN_SYM call_p3u_ri( unsigned int a, unsigned int b, unsigned int c )
{
   return (int)(a + b + c);
}


unsigned int FALCON_DYN_SYM call_p3u_ru( unsigned int a, unsigned int b, unsigned int c )
{
   return a + b + c;
}

int64 FALCON_DYN_SYM call_p3i_rl( int a, int b, int c )
{
   int64 test = 1;
   test <<= 32;
   return a + b + c + test;
}

int64 FALCON_DYN_SYM call_p3l_rl( int64 a, int64 b, int64 c )
{
   int64 test = 1;
   test <<= 32;
   return a + b + c + test;
}

float FALCON_DYN_SYM call_pfdi_rf( float x, double y, int c )
{
   return (float) y + x + c;
}

double FALCON_DYN_SYM call_pfdi_rd( float x, double y, int c )
{
   return y + x + c;
}

//===========================================
// String data
//

static char *sz_data = "Hello world";

char* FALCON_DYN_SYM call_rsz()
{
   return sz_data;
}

int FALCON_DYN_SYM call_psz_ri_check( const char *data )
{
   const char *p1 = data;
   const char *p2 = sz_data;
   while( *p1 == *p2 && *p1 != 0 )
   {
      ++p1;
      ++p2;
   }

   if (*p1 == 0 )
      return 0;

   return *p1 > *p2 ? 1 : -1;
}

//===========================================
// Wide string data
//

static wchar_t *wz_data = L"Hello world";

wchar_t* FALCON_DYN_SYM call_rsz()
{
   return wz_data;
}

int FALCON_DYN_SYM call_pwz_ri_check( const wchar_t *data )
{
   const char *p1 = data;
   const char *p2 = wz_data;

   while( *p1 == *p2 && *p1 != 0 )
   {
      ++p1;
      ++p2;
   }

   if (*p1 == 0 )
      return 0;

   return *p1 > *p2 ? 1 : -1;
}

//===========================================
// useful to check the content of memory
//

uint64 checksum( const byte *data, uint32 size )
{
   uint64 ret = 0;
   int bitmask = 1 << sizeof( uint64 );

   while ( size > 0 )
   {
      ret |= *data << bitmask;

      if ( bitmask == 0 )
         bitmask = 1 << sizeof( uint64 );
      else
         bitmask--;

      ++data;
   }

   return ret;
}

//===========================================
// Param byref tests -- int
//

void call_piiRi( int a, int b, int *sum )
{
   *sum = a + b;
}

void call_piiRu( int a, int b, unsigned int *sum )
{
   *sum = a + b;
}

void call_piiRl( int a, int b, int64 *sum )
{
   int64 v = 1 << 33;
   *sum = a + b;
}

//===========================================
// Param byref tests -- strings
//

void call_pRsz( char **sz )
{
   *sz = sz_data;
}

void call_pRwz( wchar_t **wz )
{
   *wz = wz_data;
}

