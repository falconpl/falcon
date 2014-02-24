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

inline void trace_dummyfunc( const char* )
{
}

inline void trace_dummyfunc( const char*, const void* )
{
}

#define TRACE_ON()
#define TRACE_ON_LEVEL(...)
#define TRACE_OFF

#define MESSAGE( fmt )
#define MESSAGE1( fmt )
#define MESSAGE2( fmt )
#define MESSAGE3( fmt )

#define TRACE( fmt, ... ) 
#define TRACE1( fmt, ... )
#define TRACE2( fmt, ... )
#define TRACE3( fmt, ... )

#define TRACEVAR( type, var )
#define TRACEVAR1( type, var )
#define TRACEVAR2( type, var )
#define TRACEVAR3( type, var )
#define DEBUG_ONLY(x)

#else

#include <stdio.h>
#include <falcon/setup.h>

#ifndef SRC
#define SRC __FILE__
#endif

#define TRACE_ON()  {_falcon_trace_fp = fopen( "falcon.trace", "w" ); _falcon_trace_level = 3;}
#define TRACE_ON_FILE( _fname )  {_falcon_trace_fp = fopen( _fname, "w" ); _falcon_trace_level = 3;}
#define TRACE_ON_LEVEL( _LVL )  {_falcon_trace_fp = fopen( "falcon.trace", "w" ); _falcon_trace_level = _LVL;}
#define TRACE_ON_FILE_LEVEL( _fname, _LVL ) { _falcon_trace_fp = fopen( _fname, "w" ); _falcon_trace_level = _LVL;}
#define TRACE_OFF  fclose(_falcon_trace_fp);  {_falcon_trace_fp = 0;}
#define TRACE_SET_LEVEL( _LVL ) {_falcon_trace_level = _LVL;}
#define TRACE_GET_LEVEL() (_falcon_trace_level)


#define MESSAGE( fmt ) {if( _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", SRC, __LINE__ ); fflush(_falcon_trace_fp);}
#define MESSAGE1( fmt ) {if( _falcon_trace_level >= 1 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", SRC, __LINE__ ); fflush(_falcon_trace_fp);}
#define MESSAGE2( fmt ) {if( _falcon_trace_level >= 2 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", SRC, __LINE__ ); fflush(_falcon_trace_fp);}
#define MESSAGE3( fmt ) {if( _falcon_trace_level >= 3 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", SRC, __LINE__ ); fflush(_falcon_trace_fp);}

#define TRACE( fmt, ... ) {if( _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", SRC, __LINE__, __VA_ARGS__ ); fflush(_falcon_trace_fp); fflush(_falcon_trace_fp);}
#define TRACE1( fmt, ... ) {if( _falcon_trace_level >= 1 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", SRC, __LINE__, __VA_ARGS__ ); fflush(_falcon_trace_fp);}
#define TRACE2( fmt, ... ) {if( _falcon_trace_level >= 2 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", SRC, __LINE__, __VA_ARGS__ ); fflush(_falcon_trace_fp);}
#define TRACE3( fmt, ... ) {if( _falcon_trace_level >= 3 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: " fmt "\n", SRC, __LINE__, __VA_ARGS__ ); fflush(_falcon_trace_fp);}

#define TRACEVAR( type, var ) {if( _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: %s=%" type "\n", SRC, __LINE__, #var, var ); fflush(_falcon_trace_fp);}
#define TRACEVAR1( type, var ) {if( _falcon_trace_level >= 1 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: %s=%" type "\n", SRC, __LINE__, #var, var ); fflush(_falcon_trace_fp);}
#define TRACEVAR2( type, var ) {if( _falcon_trace_level >= 2 && _falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: %s=%" type "\n", SRC, __LINE__, #var, var ); fflush(_falcon_trace_fp);}
#define TRACEVAR3( type, var ) {if( _falcon_trace_level >= 3 &&_falcon_trace_fp != 0 ) fprintf( _falcon_trace_fp, "%s:%d: %s=%" type "\n", SRC, __LINE__, #var, var ); fflush(_falcon_trace_fp);}

#define DEBUG_ONLY(x) x

extern FALCON_DYN_SYM FILE* _falcon_trace_fp;
extern FALCON_DYN_SYM int _falcon_trace_level;

#endif

#endif

/* end of trace.h */
