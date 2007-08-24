/*
   FALCON - The Falcon Programming Language.
   FILE: flc_setup.h
   $Id: setup.h,v 1.10 2007/08/09 10:57:18 jonnymind Exp $

   Setup for compilation environment and OS specific includes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom giu 6 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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

#ifdef FALCON_SYSTEM_WIN
#include <falcon/config_win.h>
#include <windows.h>
#else
#include <falcon/config.h>
#endif

//=================================
// Windows specific defines
//

#ifdef FALCON_SYSTEM_WIN
   
   //===============================
   // Compiler specific defines
   //
   
   /* Specifics for Borlandc */
   #ifdef __BORLANDC__
      #define DLL
      #define CDECL __export __stdcall
      #define FALCON_FUNC \
         extern "C" void __export __stdcall
      #define FALCON_MODULE_TYPE \
         extern "C"  ::Falcon::Module * __export __stdcall

		#ifdef FALCON_ENGINE_EXPORTS
			#define FALCON_DYN_CLASS __declspec(dllexport)
			#define FALCON_DYN_SYM __declspec(dllexport)
			#define EXTERN_TEMPLATE
		#else
			#define FALCON_DYN_CLASS __declspec(dllimport)
			#define FALCON_DYN_SYM __declspec(dllimport)
			#define EXTERN_TEMPLATE export
		#endif
   #endif

   /* Specifigs for MSVC */
   #ifdef _MSC_VER
	  #define DLL   __declspec( dllexport )
      #undef CDECL
      #define CDECL __cdecl
      #define FALCON_FUNC \
         extern "C" DLL void CDECL
      #define FALCON_MODULE_TYPE \
		 extern "C" DLL  ::Falcon::Module * CDECL
		
		#ifndef FALCON_ENGINE_STATIC
			#ifdef FALCON_ENGINE_EXPORTS
				#define FALCON_DYN_CLASS __declspec(dllexport)
				#define FALCON_DYN_SYM __declspec(dllexport)
				#define EXTERN_TEMPLATE
			#else
				#define FALCON_DYN_CLASS __declspec(dllimport)
				#define FALCON_DYN_SYM __declspec(dllimport)
				#define EXTERN_TEMPLATE export
			#endif
		#else
			#define FALCON_DYN_CLASS
			#define FALCON_DYN_SYM
		#endif
		
      /* Includes a fixed version of a broken MSVC stl header. */
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
   #endif
   
   /* Specifics for Gcc/Mingw */
   #ifdef __GNUC__
	   #define DLL
	   #ifndef CDECL
		#define CDECL
	   #endif	
	   #define FALCON_FUNC \
	      extern "C" void

	   #ifdef FALCON_ENGINE_EXPORTS
			#define FALCON_DYN_CLASS __declspec(dllexport)
			#define FALCON_DYN_SYM __declspec(dllexport)
			#define EXTERN_TEMPLATE
		#else
			#define FALCON_DYN_CLASS __declspec(dllimport)
			#define FALCON_DYN_SYM __declspec(dllimport)
			#define EXTERN_TEMPLATE export
		#endif

	   #define FALCON_MODULE_TYPE \
	      extern "C" ::Falcon::Module *
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
   #define DLL
   #define CDECL
   #define FALCON_FUNC \
      extern "C" void

   #define FALCON_DYN_CLASS
   #define FALCON_DYN_SYM
   #define EXTERN_TEMPLATE

   #define FALCON_MODULE_TYPE \
      extern "C" ::Falcon::Module *

   #define DIR_SEP_STR   "/"
   #define DIR_SEP_CHR   '/'
   #define DEFAULT_TEMP_DIR "/tmp"
   #define FALCON_SYS_EOL "\n"

#endif

#define DEFALUT_FALCON_MODULE_INIT falcon_module_init

#define FALCON_MODULE_DECL \
   FALCON_MODULE_TYPE DEFALUT_FALCON_MODULE_INIT


#endif

/* end of setup.h */
