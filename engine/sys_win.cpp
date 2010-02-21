/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: sys_win.cpp

   System specific (win) support for VM and other Falcon parts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar nov 9 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   System level support for basic and common operatios.
*/

#define SECS_IN_HOUR 3600L
#define SECS_IN_DAY  (24*SECS_IN_HOUR)
#define SECS_IN_YEAR (365 * SECS_IN_DAY)

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/sys.h>
#include <falcon/string.h>
#include <falcon/memory.h>
#include <errno.h>

#ifndef INVALID_FILE_ATTRIBUTES
#define INVALID_FILE_ATTRIBUTES ((DWORD)-1)
#endif


namespace Falcon {
namespace Sys {

static BOOL CtrlHandler(DWORD dwCtrlType)
{
    if( CTRL_C_EVENT == dwCtrlType)
    {
       // just terminate the process.
       exit(0);
    }
    return FALSE;
}

void _dummy_ctrl_c_handler()
{
    SetConsoleCtrlHandler(
      (PHANDLER_ROUTINE) CtrlHandler,
      TRUE); 
}


numeric SYSTEMTIME_TO_SECONDS( const SYSTEMTIME &st )
{
   int secsAt[12];
   secsAt[0 ] = 0;
   secsAt[1 ] = 31 * SECS_IN_DAY;
   secsAt[2 ] = secsAt[1 ] + 28 * SECS_IN_DAY;
   if( st.wYear % 4 == 0 )
      secsAt[2 ] += SECS_IN_DAY;

   secsAt[3 ] = secsAt[2 ] + 31 * SECS_IN_DAY;
   secsAt[4 ] = secsAt[3 ] + 30 * SECS_IN_DAY;
   secsAt[5 ] = secsAt[4 ] + 31 * SECS_IN_DAY;
   secsAt[6 ] = secsAt[5 ] + 30 * SECS_IN_DAY;
   secsAt[7 ] = secsAt[6 ] + 31 * SECS_IN_DAY;
   secsAt[8 ] = secsAt[7 ] + 31 * SECS_IN_DAY;
   secsAt[9 ] = secsAt[8 ] + 30 * SECS_IN_DAY;
   secsAt[10] = secsAt[9 ] + 31 * SECS_IN_DAY;
   secsAt[11] = secsAt[10] + 31 * SECS_IN_DAY;

   if( st.wYear < 1970 ) {
      int leapSeconds = (1969 - st.wYear)/4 * SECS_IN_DAY;
      return
         (1969 - st.wYear) * SECS_IN_YEAR +
         secsAt[st.wMonth-1] +
         st.wDay * SECS_IN_DAY +
         st.wHour * SECS_IN_HOUR +
         st.wMinute * 60 +
         st.wSecond +
         leapSeconds +
         (st.wMilliseconds / 1000.0);
   }
   else {
      // good also if wYear is 1970: /4 will neutralize it.
      int leapSeconds = ((st.wYear-1)-1970)/4 * SECS_IN_DAY;

      return
         (st.wYear-1970 ) * SECS_IN_YEAR +
         secsAt[st.wMonth-1] +
         st.wDay * SECS_IN_DAY +
         st.wHour * SECS_IN_HOUR +
         st.wMinute * 60 +
         st.wSecond +
         leapSeconds +
         (st.wMilliseconds / 1000.0);
   }
}


void _sleep( numeric time )
{
   Sleep( long( time * 1000 ) );
}

numeric _seconds()
{
   SYSTEMTIME st;

   GetSystemTime( &st );
   return SYSTEMTIME_TO_SECONDS( st );
}

numeric _localSeconds()
{
   SYSTEMTIME st;

   GetLocalTime( &st );
   return SYSTEMTIME_TO_SECONDS( st );
}

uint32 _milliseconds()
{
   return (uint32) GetTickCount();
}

void _tempName( String &res )
{
   String temp_dir;
   res.bufferize();
	res.size(0);

   if ( ! Sys::_getEnv( "TEMP", temp_dir ) )
      if ( ! Sys::_getEnv( "TMP", temp_dir ) )
         temp_dir = "C:\\TEMP";

   int tempLen = temp_dir.length() * sizeof( wchar_t ) + sizeof( wchar_t );
   wchar_t *wct = (wchar_t *) memAlloc( tempLen );
   temp_dir.toWideString( wct, tempLen );

   DWORD attribs = GetFileAttributesW( wct );
   if( attribs == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		char *bufname_c = (char *) wct;
		if( temp_dir.toCString( bufname_c, tempLen ) > 0 )
			attribs = GetFileAttributesA( bufname_c );
	}

   if ( GetLastError() != 0 || ((attribs & FILE_ATTRIBUTE_DIRECTORY) == 0) )
   {
      temp_dir = ".";
   }
   memFree( wct );

   res += temp_dir;
	res += "\\";
	res += "falcon_tmp_";
	res.writeNumberHex( GetCurrentProcessId() );
   /*res += "_";
   res.writeNumberHex( (uint32) rand() );*/
}

int64 _lastError()
{
   return (int64) GetLastError();
}


bool _describeError( int64 lastError, String &dest )
{
   LPVOID lpMsgBuf;
   DWORD error = (DWORD) lastError;

   DWORD res = FormatMessage(
      FORMAT_MESSAGE_ALLOCATE_BUFFER |
      FORMAT_MESSAGE_FROM_SYSTEM,
      0,
      error,
      LANG_USER_DEFAULT,
      (LPTSTR) &lpMsgBuf,
      0,
      NULL
    );

   if ( res == 0 ) {
      dest = "Impossible to retreive error description";
		dest.bufferize();
      return false;
   }
   else
   {
      dest = (char *) lpMsgBuf;
		dest.bufferize();
   }

   LocalFree(lpMsgBuf);
   return true;
}

bool _getEnv( const String &var, String &result )
{

	#if _MSC_VER < 1400
		char convertBuf[1024];
		if ( var.toCString( convertBuf, 1024 ) )
		{
			char *value = getenv( convertBuf );
			if( value != 0 )
			{
				result.bufferize( value );
				return true;
			}
		}

	#else
	   wchar_t convertBuf[512]; // system var names larger than 512 are crazy.
		if ( var.toWideString( convertBuf, 512 ) )
		{

			wchar_t *value = (wchar_t *) memAlloc( 512 * sizeof( wchar_t ) );
			size_t retSize;
			errno_t error = _wgetenv_s( &retSize, value, 512, convertBuf );
			if ( error == ERANGE )
			{
				memFree( value );
				value = (wchar_t *) memAlloc( retSize * sizeof( wchar_t ) );
				error = _wgetenv_s( &retSize, value, retSize, convertBuf );
			}

			if ( error != EINVAL && retSize != 0 )
			{
				result.bufferize( value );
				memFree( value );
				return true;
			}
			memFree( value );
		}
	#endif

   return false;
}

bool _setEnv( const String &var, const String &value )
{
   #if _MSC_VER < 1400
		String temp = var + "=" + value;
		uint32 tempLen = temp.length() * 4 + 4;
		char *tempBuf = (char*) memAlloc( tempLen );
		temp.toCString( tempBuf, tempLen );
		putenv( tempBuf );
		memFree( tempBuf );
		return true;
	#else
		uint32 varLen = var.length() * sizeof(wchar_t) + sizeof(wchar_t);
		uint32 valueLen = value.length() * sizeof(wchar_t) + sizeof(wchar_t);
		wchar_t *varBuf = (wchar_t *) memAlloc( varLen );
		wchar_t *valueBuf = (wchar_t *) memAlloc( valueLen );

		var.toWideString( varBuf, varLen );
		value.toWideString( valueBuf, valueLen );

		bool result = _wputenv_s( varBuf, valueBuf ) == 0;

		memFree( varBuf );
		memFree( valueBuf );
	   return result;
	#endif
}

bool _unsetEnv( const String &var )
{
	#if _MSC_VER < 1400
		String temp = var + "=";
		uint32 tempLen = temp.length() * 4 + 4;
		char *tempBuf = (char*) memAlloc( tempLen );
		temp.toCString( tempBuf, tempLen );
		putenv( tempBuf );
		memFree( tempBuf );
		return true;
	#else
		uint32 varLen = var.length() * sizeof(wchar_t) + sizeof(wchar_t);
		wchar_t *varBuf = (wchar_t *) memAlloc( varLen );

		var.toWideString( varBuf, varLen );

		bool result = _wputenv_s( varBuf, L"" ) == 0;
		memFree( varBuf );
		return result;
	#endif
}

void _enumerateEnvironment( EnvStringCallback cb, void* cbData )
{
   #if _MSC_VER < 1400
      char* envstr = GetEnvironmentStringsA();
   #else
      wchar_t* envstr = GetEnvironmentStringsW();
   #endif

   uint32 pos = 0;
   uint32 posn = 0;
   while( envstr[posn] != 0 )
   {
      // not an error, we check it twice.
      uint32 poseq = 0;
      while( envstr[posn] != 0 )
      {
        if( poseq == 0 && envstr[posn] == '=' )
           poseq = posn;
        ++posn;
      }

      // did we find a variable?
      if( poseq != 0 )
      {
         String key, value;
         key.adopt( envstr + pos, poseq-pos, 0 );
         value.adopt( envstr + poseq+1, posn-poseq-1, 0 );
         
         key.bufferize();
         value.bufferize();

         cb( key, value, cbData );
      }

      // advancing to the next string; if the first char is zero, we exit
      ++posn;
      pos = posn;
   }

   #if _MSC_VER < 1400
      FreeEnvironmentStringsA( envstr );
   #else
      FreeEnvironmentStringsW( envstr );
   #endif
}

}
}


/* end of sys_win.cpp */
