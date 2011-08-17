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

namespace Falcon {

// for pointers
template<typename rtype_ptr, typename stype>
inline rtype_ptr dyncast(stype* pSource)
{
#ifndef NDEBUG
   // Fassert should resolve in nothing in release, but it may change in future.
   fassert ( pSource == 0 || ( static_cast<rtype_ptr>(pSource) == dynamic_cast<rtype_ptr>(pSource) ) );
#endif

   return static_cast<rtype_ptr>(pSource);
}

// for references
/* Breaks MINGW
template<typename rtype_ref, typename stype>
inline rtype_ref dyncast(stype& rSource)
{
#ifndef NDEBUG
   try
   {
     fassert ( &static_cast<rtype_ref>(rSource) == &dynamic_cast<rtype_ref>(rSource) );
   }
   catch(...)
   {
     // Block exceptions from dynamic_cast and assert instead.
     fassert(false);
   }
#endif
   return static_cast<rtype_ref>(rSource);
}
*/
}

#endif

/* end of fassert.h */
