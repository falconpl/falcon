/*
   FALCON - The Falcon Programming Language.
   FILE: fassert.h

   Falcon specific assertion raising
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 4 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon specific assertion raising
*/

#ifndef flc_fassert_H
#define flc_fassert_H

#include <falcon/setup.h>

#ifndef NDEBUG
	#ifdef _MSC_VER
		#ifdef __func__
			# define fassert(expr) \
			{if (!(expr)) _perform_FALCON_assert_func( #expr, __FILE__, __LINE__, __func__ );}
		#else
			# define fassert(expr) \
				{if (!(expr)) _perform_FALCON_assert( #expr, __FILE__, __LINE__ );}
		#endif
   #else

      // older versions of g++/mingw hadn't __STRING
      #ifndef __STRING
      #define __STRING(x) #x
      #endif

		#ifdef __func__
			# define fassert(expr) \
			{if (!(expr)) _perform_FALCON_assert_func( __STRING(expr), __FILE__, __LINE__, __func__ );}
		#else
			# define fassert(expr) \
				{if (!(expr)) _perform_FALCON_assert( __STRING(expr), __FILE__, __LINE__ );}
		#endif
   #endif
#else
   # define fassert(expr)
#endif

extern "C" void FALCON_DYN_SYM _perform_FALCON_assert_func( const char *expr, const char *filename, int line, const char *assertFunc );
extern "C" void FALCON_DYN_SYM _perform_FALCON_assert( const char *expr, const char *filename, int line );

#endif

/* end of fassert.h */
