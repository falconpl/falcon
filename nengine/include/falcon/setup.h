/*
   FALCON - The Falcon Programming Language.
   FILE: flc_setup.h

   Setup for compilation environment and OS specific includes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom giu 6 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Setup for compilation environment and OS specific includes.

This file is responsible for inclusion of all the system or compiler
specific resources. It should be directly or indirectly included by
all the falcon inclusion files.

It should be included also (and mainly by) extension libraries, that
will find here the macros needed to create dynamic link libraries under
different systems.
*/


#ifndef FLC_SETUP_H
#define FLC_SETUP_H

#include <falcon/config.h>

//=================================
// Windows specific defines
//

#ifdef FALCON_SYSTEM_WIN
   
   // Minimal specific.
   #if ! defined(_WIN32_WINNT)
   #define _WIN32_WINNT 0x0403
   #endif

   //===============================
   // Compiler specific defines
   //


   /* Specifigs for MSVC */
   #ifdef _MSC_VER
      #undef CDECL
      #define CDECL __cdecl
      #define FALCON_FUNC \
         void CDECL

		#ifndef FALCON_ENGINE_STATIC
			#if defined(FALCON_ENGINE_EXPORTS)
				#define FALCON_DYN_CLASS __declspec(dllexport)
				#define FALCON_DYN_SYM __declspec(dllexport)
				#define EXTERN_TEMPLATE
            
            // Falcon export service is optional, but mandatory with engine exports.
            #ifndef FALCON_EXPORT_SERVICE
               #define FALCON_EXPORT_SERVICE
            #endif
			#else
				#define FALCON_DYN_CLASS __declspec(dllimport)
				#define FALCON_DYN_SYM __declspec(dllimport)
				#define EXTERN_TEMPLATE export
			#endif
		#else
			#define FALCON_DYN_CLASS
			#define FALCON_DYN_SYM
		#endif

      #ifdef FALCON_EXPORT_SERVICE
         #define FALCON_SERVICE __declspec(dllexport)
      #else
         #define FALCON_SERVICE __declspec(dllimport)
      #endif

      #define FALCON_FUNC_DYN_SYM \
		   FALCON_DYN_SYM void CDECL

      #define FALCON_MODULE_TYPE \
		   extern "C" __declspec(dllexport) ::Falcon::Module * CDECL


      #pragma warning (disable: 4786 )
      #pragma warning (disable: 4291 )
      #pragma warning (disable: 579 )
      #pragma warning (disable: 4290 )
      #pragma warning (disable: 4231 )
      #pragma warning (disable: 4355)
      #pragma warning (disable: 4996)

   	#if _MSC_VER <= 1400
   		#ifndef _WCHAR_T_DEFINED
   			typedef unsigned short wchar_t;
   		#endif
   	#endif

      #define atoll _atoi64
      #define snprintf _snprintf
      #define LLFMT    "I64"
      #define I64LIT(x) (x ## i64)
      #define UI64LIT(x) (x ## ui64)
   #endif

   /* Specifics for Gcc/Mingw */
   #ifdef __GNUC__
	   #ifndef CDECL
		#define CDECL
	   #endif
	   #define FALCON_FUNC \
	      void

	   #ifdef FALCON_ENGINE_EXPORTS
			#define FALCON_DYN_CLASS __declspec(dllexport)
			#define FALCON_DYN_SYM __declspec(dllexport)
			#define EXTERN_TEMPLATE
		#else
			#define FALCON_DYN_CLASS __declspec(dllimport)
			#define FALCON_DYN_SYM __declspec(dllimport)
			#define EXTERN_TEMPLATE export
		#endif

      #ifdef FALCON_EXPORT_SERVICE
         #define FALCON_SERVICE __declspec(dllexport)
      #else
         #define FALCON_SERVICE __declspec(dllimport)
      #endif

      #define FALCON_FUNC_DYN_SYM \
		   FALCON_DYN_SYM void CDECL

	   #define FALCON_MODULE_TYPE \
	      extern "C" __declspec(dllexport) ::Falcon::Module *
      #define LLFMT    "ll"
      #define I64LIT(x) (x ## LL)
      #define UI64LIT(x) (x ## ULL)
	#endif

   /* Other Windonws specific system defines */

   #define DIR_SEP_STR   "\\"
   #define DIR_SEP_CHR   '\\'
   // paths are always indicated in falcon convention.
   #define DEFAULT_TEMP_DIR "C:/TEMP"
   #define FALCON_SYS_EOL "\r\n"

//=================================
// Unix specific defines
//
#else
   #define CDECL 
   #define FALCON_FUNC \
      void

   #define FALCON_DYN_CLASS
   #define FALCON_DYN_SYM 
   #define EXTERN_TEMPLATE
   #define FALCON_SERVICE

   #define FALCON_MODULE_TYPE \
      extern "C" ::Falcon::Module *

   #define FALCON_FUNC_DYN_SYM   FALCON_FUNC

   #define DIR_SEP_STR   "/"
   #define DIR_SEP_CHR   '/'
   #define DEFAULT_TEMP_DIR "/tmp"
   #define FALCON_SYS_EOL "\n"
   #define LLFMT "ll"
   #define I64LIT(x) (x ## LL)
   #define UI64LIT(x) (x ## ULL)

#endif

//===================================
// Helper STR / _STR / #x converter
//
#ifndef _STR
   #define _STR(x) #x
#endif

#ifndef STR
   #define STR(x) _STR(x)
#endif

#endif

/* end of setup.h */
