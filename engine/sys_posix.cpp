/*
   FALCON - The Falcon Programming Language.
   FILE: sys_posix.cpp

   System specific (unix) support for VM.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar nov 9 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#ifdef __APPLE__
#include <crt_externs.h>
#define environ (*_NSGetEnviron())
#else
extern "C"
{
   extern char **environ;
}
#endif

#include <unistd.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/sys.h>
#include <falcon/string.h>
//#include <falcon/transcoding.h>

namespace Falcon {
namespace Sys {

void _dummy_ctrl_c_handler()
{
}

numeric _seconds()
{
   struct timeval time;
   gettimeofday( &time, 0 );
   return time.tv_sec + (time.tv_usec / 1000000.0 );
}


numeric _localSeconds()
{
   struct timeval current;
   struct tm date_local, date_gm;
   time_t t;

   gettimeofday( &current, 0 );
   time( &t );
   localtime_r( &t, &date_local );
   gmtime_r( &t, &date_gm );
   time_t leap = mktime( &date_local) - mktime( &date_gm );

   return leap + current.tv_sec + (current.tv_usec / 1000000.0 );
}

int64 _milliseconds()
{
#if POSIX_TIMERS > 0
   struct timespec time;
   clock_gettime( CLOCK_REALTIME, &time );
   int64 msecs = time.tv_sec;
   msecs *= 1000;
   msecs += time.tv_nsec / 1000000;
   return msecs;
#else
   struct timeval time;
   gettimeofday( &time, 0 );
   int64 msecs = time.tv_sec;
   msecs *= 1000;
   msecs += time.tv_usec / 1000;
   return msecs;
#endif
}

int64 _epoch()
{
   return (int64) time(0);
}

void _tempName( String &res )
{
   static bool first = true;
   const char *temp_dir;
   struct stat st;

   if( first ) {
      first = false;
      time_t t;
      srand( (unsigned int ) time( &t ) );
   }

   temp_dir = getenv( "TMP" );

   if ( temp_dir == 0 )
      temp_dir = getenv( "TMPDIR" );

   if ( temp_dir == 0 ) {
      temp_dir = DEFAULT_TEMP_DIR;
   }

   if ( stat( temp_dir, &st ) == -1 || ! S_ISDIR( st.st_mode ) ) {
      temp_dir = ".";
   }

   res = temp_dir;
   res.append( "/falcon_tmp_" );
   res.writeNumber( (int64) getpid() );
   res.append("_");
   res.writeNumber( (int64) rand() );
   res.bufferize();
   // force buffering
}

bool _describeError( int64 eid, String &target )
{
   const char *error = strerror( eid );
   if( error != 0 ) {
      target.bufferize( error );
      return true;
   }

   return false;
}

int64 _lastError()
{
   return (int64) errno;
}

bool _getEnv( const String &var, String &result )
{
   static char convertBuf[512]; // system var names larger than 512 are crazy.
   // in unix system, we have at worst UTF-8 var names.
   if ( var.toCString( convertBuf, 512 ) != String::npos )
   {
      char *value = getenv( convertBuf );
      if ( value != 0 )
      {
         result.fromUTF8( value );
         return true;
      }
   }

   return false;
}

bool _setEnv( const String &var, const String &value )
{
   // in unix system, we have at worst UTF-8 var names.
   uint32 varLen = var.length() * 4 + 2;
   uint32 valueLen = value.length() * 4 + 2;
   char *varBuf = (char *) malloc( varLen );
   char *valueBuf = (char *) malloc( valueLen );

   var.toCString( varBuf, varLen );
   value.toCString( valueBuf, valueLen );

   bool result = setenv( varBuf, valueBuf, 1 ) == 0;
   free( varBuf );
   free( valueBuf );
   return result;
}

bool _unsetEnv( const String &var )
{
   // in unix system, we have at worst UTF-8 var names.
   uint32 varLen = var.length() * 4 + 2;
   char *varBuf = (char *) malloc( varLen );

   var.toCString( varBuf, varLen );

   /* currently unsetenv does not return in darwin;
      we need sometime to find a solution
   bool result = unsetenv( varBuf ) == 0;
   */
   bool result = true;
   unsetenv( varBuf );
   free( varBuf );
   return result;
}

void _enumerateEnvironment( EnvStringCallback cb, void* cbData )
{
   // do we know which encoding are we using?
   String enc;

   char** env = environ;
   while( *env != 0 )
   {
      String temp;
      temp = *env;

      uint32 pos;
      if ( (pos = temp.find( '=' )) != String::npos )
      {
         cb( temp.subString(0,pos), temp.subString(pos+1), cbData );
      }

      ++env;
   }
}

/*
void _enumerateEnvironment( EnvStringCallback cb, void* cbData )
{
   // do we know which encoding are we using?
   String enc;
   bool bTranscode = GetSystemEncoding( enc ) && enc != "C";

   char** env = environ;
   while( *env != 0 )
   {
      String temp;
      if( bTranscode )
      {
         if( ! TranscodeFromString( *env, enc, temp ) )
         {
            bTranscode = false;
            temp = *env;
         }
      }
      else
         temp = *env;

      uint32 pos;
      if ( (pos = temp.find( '=' )) != String::npos )
      {
         cb( temp.subString(0,pos), temp.subString(pos+1), cbData );
      }

      ++env;
   }
}
*/

int64 _getpid() {
   return (int64) getpid();
}

long _getPageSize()
{
   #ifdef _SC_PAGESIZE
   return sysconf( _SC_PAGESIZE );
   #else
   return (long) getpagesize();
   #endif
}


int _getCores()
{
   return sysconf( _SC_NPROCESSORS_ONLN );
}

bool _getCWD( String& name )
{
   char buf[PATH_MAX+1];
   char* res = getcwd( buf, PATH_MAX );
   if( res == 0 )
   {
      return false;
   }
   
   name.fromUTF8( buf );
   return true;
}
}
}

/* end of sys_posix.cpp */
