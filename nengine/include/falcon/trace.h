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

#define TRACE_ON()
#define TRACE_ON_LEVEL(...)
#define TRACE_OFF

#define MESSAGE(...)
#define MESSAGE1(...)

#define TRACE(...)
#define TRACE1(...)
#define TRACE2(...)
#define TRACE3(...)

#define TRACEVAR(...)
#define TRACEVAR1(...)
#define TRACEVAR2(...)
#define TRACEVAR3(...)

#else

#include <stdio.h>
#include <falcon/setup.h>

#define TRACE_ON()  _falcon_trace_fp = fopen( "falcon.trace", "w" ); _falcon_trace_level = 3;
#define TRACE_ON_LEVEL( _LVL )  _falcon_trace_fp = fopen( "falcon.trace", "w" ); _falcon_trace_level = _LVL;
#define TRACE_OFF  fclose(_falcon_trace_fp);  _falcon_trace_fp = 0;

#define MESSAGE( fmt ) if( _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", __FILE__, __LINE__ ); fflush(_falcon_trace_fp);
#define MESSAGE1( fmt ) if( _falcon_trace_level >= 1 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", __FILE__, __LINE__ ); fflush(_falcon_trace_fp);

#define TRACE( fmt, ... ) if( _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", __FILE__, __LINE__, __VA_ARGS__ ); fflush(_falcon_trace_fp); fflush(_falcon_trace_fp);
#define TRACE1( fmt, ... ) if( _falcon_trace_level >= 1 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", __FILE__, __LINE__, __VA_ARGS__ ); fflush(_falcon_trace_fp);
#define TRACE2( fmt, ... ) if( _falcon_trace_level >= 2 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", __FILE__, __LINE__, __VA_ARGS__ ); fflush(_falcon_trace_fp);
#define TRACE3( fmt, ... ) if( _falcon_trace_level >= 3 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", __FILE__, __LINE__, __VA_ARGS__ ); fflush(_falcon_trace_fp);

#define TRACEVAR( type, var ) if( _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: %s=%" type "\n", __FILE__, __LINE__, #var, var ); fflush(_falcon_trace_fp);
#define TRACEVAR1( type, var ) if( _falcon_trace_level >= 1 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: %s=%" type "\n", __FILE__, __LINE__, #var, var ); fflush(_falcon_trace_fp);
#define TRACEVAR2( type, var ) if( _falcon_trace_level >= 2 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: %s=%" type "\n", __FILE__, __LINE__, #var, var ); fflush(_falcon_trace_fp);
#define TRACEVAR3( type, var ) if( _falcon_trace_level >= 3 &&_falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: %s=%" type "\n", __FILE__, __LINE__, #var, var ); fflush(_falcon_trace_fp);

extern FALCON_DYN_SYM FILE* _falcon_trace_fp;
extern FALCON_DYN_SYM int _falcon_trace_level;

#endif

#endif

/* end of trace.h */
