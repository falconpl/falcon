/*
   FALCON - The Falcon Programming Language.
   FILE: process_sys_unix.cpp

   Unix specific implementation of openProcess
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun Jan 30 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Unix specific implementation of openProcess
*/

#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>

#include <falcon/memory.h>
#include <falcon/fstream_sys_unix.h>

#include "process_sys_unix.h"

#include <string.h>

namespace Falcon {

namespace Sys {

static char **s_localize( String **args )
{
   char **argv;

   uint32 count = 0;
   while( args[count] != 0 )
      ++count;

   argv = (char **) memAlloc( (count+1) *sizeof( char * ) );
   argv[ count ] = 0;
   count = 0;
   while( args[count] != 0 )
   {
      String *arg = args[count];
      uint32 allocSize = arg->length() * 4;
      char *buffer = (char *) memAlloc( allocSize );
      arg->toCString( buffer, allocSize );
      argv[ count ] = buffer;
      ++count;
   }

   return argv;
}

static void s_freeLocalized( char **args )
{
   uint32 count = 0;
   while( args[ count ] != 0 )
   {
      memFree( args[count] );
      ++count;
   }
   memFree( args );

}

//====================================================================
// Simple process manipulation functions

uint64 processId()
{
   return (uint64) getpid();
}

bool processKill( uint64 id )
{
   return kill( (pid_t) id, SIGKILL ) == 0;
}

bool processTerminate( uint64 id )
{
   return (int64) kill( (pid_t) id, SIGTERM ) == 0;
}

//====================================================================
// Process enumerator
ProcessEnum::ProcessEnum()
{
   m_sysdata = opendir( "/proc" );
}

ProcessEnum::~ProcessEnum()
{
   close();
}

int ProcessEnum::next( String &name, uint64 &pid, uint64 &ppid, String &path )
{
   if ( m_sysdata == 0 )
      return -1;

   DIR *procdir = (DIR *) m_sysdata;
   struct dirent *de;

   while ( (de = readdir( procdir ) ) != 0 )
   {
      if ( de->d_name[0] >= '0' && de->d_name[0] <= '9' )
         break;
   }

   if ( de == 0 )
      return 0;

   char statent[ 64 ];
   FILE *fp;
   char status;
   char szName[1024];

   snprintf( statent, 64, "/proc/%s/stat", de->d_name );
   fp = fopen( statent, "r" );
   if ( fp == NULL ) return -1;
   int32 p_pid, p_ppid;
   if ( fscanf( fp, "%d %s %c %d", &p_pid, szName, &status, &p_ppid ) != 4 )
   {
      fclose( fp );
      return -1;
   }
   pid = (int64) p_pid;
   ppid = (int64) p_ppid;
   fclose(fp);

   if ( szName[0] == '(' )
   {
      szName[ strlen( szName ) -1] = '\0';
      name.bufferize( szName + 1 );
   }
   else {
      name.bufferize( szName );
   }

   // read also the command line, which may be missing.
   snprintf( statent, 255, "/proc/%s/cmdline", de->d_name );
   fp = fopen( statent, "r" );
   if ( fp == NULL || fscanf( fp, "%s", szName ) != 1 )
   {
      szName[0] = 0;
      return 1;
   }

   fclose(fp);
   path.bufferize( szName );

   return 1;
}

bool ProcessEnum::close()
{
   if ( m_sysdata != 0 )
   {
      closedir( (DIR *) m_sysdata );
      m_sysdata = 0;
      return true;
   }
   return false;
}


//====================================================================
// Generic system interface.

bool spawn( String **args, bool overlay, bool background, int *returnValue )
{
   // convert to our local format.
   char **argv = s_localize( args );

   if ( ! overlay )
   {
      pid_t pid = fork();

      if ( pid == 0 ) {
         // we are in the child;
         if ( background ) {
            // if child output is not wanted, sink it
            int hNull;
            hNull = open("/dev/null", O_RDWR);
            dup2( hNull, 0 );
            dup2( hNull, 1 );
            dup2( hNull, 2 );
         }

         execvp( argv[0], argv ); // never returns.
         _exit( -1 ); // or we have an error
      }

      s_freeLocalized( argv );
      if ( pid == waitpid( pid, returnValue, 0 ) )
         return true;
      // else we have an error
      *returnValue = errno;
      return false;
   }

   // in case of overlay, just run the execvp and eventually return in case of error.
   execvp( argv[0], argv ); // never returns.
   _exit( -1 );
}

const char *shellName()
{
   char *shname = getenv("SHELL");
   if ( shname == 0 )
      shname = "/bin/sh";
   return shname;
}

const char *shellParam()
{
   return "-c";
}

ProcessHandle *openProcess( String **arg_list, bool sinkin, bool sinkout, bool sinkerr, bool mergeErr, bool bg )
{
   UnixProcessHandle *ph = new UnixProcessHandle();

   // step 1: prepare the needed pipes
   if ( sinkin )
      ph->m_file_des_in[1] = -1;
   else
   {
      if ( pipe( ph->m_file_des_in ) < 0 ) {
         ph->lastError(errno);
         return ph;
      }
   }

   if ( sinkout )
      ph->m_file_des_out[0] = -1;
   else
   {
      if ( pipe( ph->m_file_des_out ) < 0 ) {
         ph->lastError(errno);
         return ph;
      }
   }

   if ( sinkerr )
      ph->m_file_des_err[0] = -1;
   else if ( mergeErr )
      ph->m_file_des_err[0] = ph->m_file_des_out[0];
   else
   {
      if ( pipe( ph->m_file_des_err ) < 0 ) {
         ph->lastError(errno);
         return ph;
      }
   }

   //Second step: fork
   ph->m_pid = fork();

   if ( ph->m_pid == 0 )
   {
      int hNull;
      // do we need to sink?
      if ( sinkin || sinkout || sinkerr )
         hNull = open("/dev/null", O_RDWR);

      // Third step: prepare the streams
      if ( sinkin )
         dup2( hNull, STDIN_FILENO );
      else
         dup2( ph->m_file_des_in[0], STDIN_FILENO );

      if ( sinkout )
         dup2( hNull, STDOUT_FILENO);
      else
         dup2( ph->m_file_des_out[1], STDOUT_FILENO );

      if( sinkerr )
         dup2( hNull, STDERR_FILENO );
      else if( mergeErr )
         dup2( ph->m_file_des_out[1], STDERR_FILENO );
      else
         dup2( ph->m_file_des_err[1], STDERR_FILENO );

      // Launch the EXECVP procedure.
      char **args = s_localize( arg_list );
      execvp( args[0], args ); // never returns.
      _exit( -1 );
   }
   else
      return ph;
}

//====================================================================
// UnixProcessHandle system area.

UnixProcessHandle::~UnixProcessHandle()
{
   if ( ! done() )
   {
      close();
      terminate( true );
      wait( true );
   }
}

bool UnixProcessHandle::wait( bool block )
{
   int pval,res;
   res = waitpid( m_pid, &pval, block ? 0 : WNOHANG );
   if( res == m_pid )
   {
      done(true);
      processValue( WEXITSTATUS(pval) );

      return true;
   }
   else  if( res == 0 ) {
      done(false);
      return true;
   }
   lastError( errno );
   return false;
}


bool UnixProcessHandle::close()
{
   if ( m_file_des_err[1] != -1 )
      ::close(m_file_des_err[1]);
   if ( m_file_des_out[0] != -1 )
      ::close( m_file_des_out[0] );
   if ( m_file_des_err[0] != -1 )
      ::close( m_file_des_err[0] );
   return true;
}

bool UnixProcessHandle::terminate( bool severe )
{
   int sig = severe ? SIGKILL : SIGTERM;
   if( kill( m_pid, sig ) == 0) {
      return true;
   }
   lastError( errno );
   return false;
}

::Falcon::Stream *UnixProcessHandle::getInputStream()
{
   if( m_file_des_in[1] == -1 || done() )
      return 0;

   UnixFileSysData *data = new UnixFileSysData( m_file_des_in[1], 0 );
   return new FileStream( data );
}

::Falcon::Stream *UnixProcessHandle::getOutputStream()
{
   if( m_file_des_out[0] == -1 || done() )
      return 0;

   UnixFileSysData *data = new UnixFileSysData( m_file_des_out[0], 0 );
   return new FileStream( data );
}

::Falcon::Stream *UnixProcessHandle::getErrorStream()
{
   if( m_file_des_err[0] != -1 || done() )
      return 0;

   UnixFileSysData *data = new UnixFileSysData( m_file_des_err[0], 0 );
   return new FileStream( data );

}

}
}

/* end of process_sys_unix.cpp */
