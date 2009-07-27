/*
   FALCON - The Falcon Programming Language.
   FILE: trace.h

   Debug trace utility.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 18 Jul 2009 14:07:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_TRACE_H_
#define _FALCON_TRACE_H_

#ifdef NDEBUG

#define TRACE_ON(...)
#define TRACE_OFF
#define TRACE(...)
#define MESSAGE(...)

#else

#include <stdio.h>
#include <falcon/setup.h>

#define TRACE_ON( name )  _falcon_trace_fp = fopen( name, "w" );
#define TRACE_OFF  fclose(_falcon_trace_fp);  _falcon_trace_fp = 0;

#define TRACE( fmt, ... ) if( _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", __FILE__, __LINE__, __VA_ARGS__ );
#define MESSAGE( fmt ) if( _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", __FILE__, __LINE__ );
#define TRACEVAR( type, var ) if( _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: %s=%" type "\n", __FILE__, __LINE__, #var, var );

extern FALCON_DYN_SYM FILE* _falcon_trace_fp;

#endif

#endif

/* end of trace.h */
