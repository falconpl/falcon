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

#include "process_posix.h"

#include <string.h>

namespace Falcon { namespace Sys {


namespace {

struct LocalizedArgv
{
   char** p;

   LocalizedArgv(String** argList) :
      p( 0 )
   {
      this->fill( argList );
   }

    ~LocalizedArgv()
    {
       this->free();
    }

   void fill( String** argList )
   {
      this->free();

      size_t size = 0;
      while( argList[size] != 0 )
         ++size;

      p = new char*[size + 1];
      p[size] = 0;

      for(size_t i = 0; argList[i] != 0; i++ )
      {
         String* arg = argList[i];
         size_t nBytes = arg->length() * 4;
         p[i] = new char[ nBytes ];
         arg->toCString(p[i], nBytes );
      }
   }

   void free()
   {
      if( !p ) return;

      for(size_t i = 0; p[i] != 0; i++ )
         delete [] p[i];
      delete [] p;
   }
};

} // anonymous namespace

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
   this->close();
}

int ProcessEnum::next( String &name, uint64 &pid, uint64 &ppid, String &commandLine )
{
   if ( m_sysdata == 0 )
      return -1;

   DIR* procdir = static_cast<DIR*>(m_sysdata);
   struct dirent* de;

   // `DIR' implements a stream and readdir moves it forward.
   while ( (de = readdir( procdir ) ) != 0 )
   {
      // skip non pid entries
      if ( de->d_name[0] >= '0' && de->d_name[0] <= '9' )
         break;
   }
   if ( !de ) return 0; // EOF
   
   char statent[ 256 ];
   snprintf( statent, 256, "/proc/%s/stat", de->d_name );
   FILE* fp = fopen( statent, "r" );
   if ( !fp ) return -1;
   
   int32 p_pid, p_ppid;
   char status;
   char szName[1024];
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
   else
      name.bufferize( szName );

   // read also the command line, which may be missing.
   snprintf( statent, sizeof(statent), "/proc/%s/cmdline", de->d_name );
   fp = fopen( statent, "r" );
   if ( !fp || fscanf( fp, "%s", szName ) != 1 )
   {
      szName[0] = 0;
      return 1;
   }

   fclose( fp );
   commandLine.bufferize( szName );

   return 1;
}

bool ProcessEnum::close()
{
   if ( m_sysdata )
   {
      closedir( static_cast<DIR*>(m_sysdata) );
      m_sysdata = 0;
      return true;
   }
   return false;
}


//====================================================================
// Generic system interface.

bool spawn( String** argList, bool overlay, bool background, int* returnValue )
{
   // convert to our local format.
   LocalizedArgv argv( argList );

   if ( ! overlay )
   {
      pid_t pid = fork();

      if ( !pid )
      {
         // we are in the child;
         if ( background )
         {
            // if child output is not wanted, sink it
            int hNull;
            hNull = open("/dev/null", O_RDWR);

            dup2( hNull, STDIN_FILENO );
            dup2( hNull, STDOUT_FILENO );
            dup2( hNull, STDERR_FILENO );
         }

         execvp( argv.p[0], argv.p ); // never returns.
         exit( -1 ); // or we have an error
      }

      if ( pid == waitpid( pid, returnValue, 0 ) )
         return true;
      // else we have an error
      *returnValue = errno;
      return false;
   }

   // in case of overlay, just run the execvp and eventually return in case of error.
   execvp( argv.p[0], argv.p ); // never returns.
   exit( -1 );
}



bool spawn_read( String** argList, bool overlay, bool background, int* returnValue, String* sOutput )
{
   int pipe_fd[2];

   if ( pipe( pipe_fd ) != 0 )
      return false;

   // convert to our local format.
   LocalizedArgv argv( argList );
   const char* cookie = "---ASKasdfyug72348AIOfasdjkfb---";

   if ( ! overlay )
   {
      pid_t pid = fork();

      if ( pid == 0 )
      {
         // we are in the child;
         if ( background )
         {
            // if child output is not wanted, sink it
            int hNull;
            hNull = open("/dev/null", O_RDWR);

            dup2( hNull, STDIN_FILENO );
            dup2( hNull, STDERR_FILENO );
         }

         dup2( pipe_fd[1], STDOUT_FILENO );

         execvp( argv.p[0], argv.p ); // never returns.
         write( pipe_fd[1], cookie, strlen( cookie ) );
         exit( -1 ); // or we have an error
      }

      // read the output
      const size_t max_read_per_loop = 4096;
      char buffer[max_read_per_loop];
      int readin;
      fd_set rfds;
      struct timeval tv;

      /* Wait up to 100msecs */
      tv.tv_sec = 0;
      tv.tv_usec = 100;

      while( true )
      {
         FD_ZERO( &rfds );
         FD_SET( pipe_fd[0], &rfds);
         int retval = select(pipe_fd[0]+1, &rfds, NULL, NULL, &tv );

         if( retval )
         {
            readin = read( pipe_fd[0], buffer, max_read_per_loop );
            String s;
            s.adopt( buffer, readin, 0 );
            sOutput->append( s );
         }
         else
         {
            if ( pid == waitpid( pid, returnValue, WNOHANG ) )
            {
               close( pipe_fd[0] );
               close( pipe_fd[1] );
               return *sOutput != cookie;
            }
         }
      }
   }

   // in case of overlay, just run the execvp and eventually return in case of error.
   execvp( argv.p[0], argv.p ); // never returns..
   // .. unless an error occured:
   exit( -1 );
   // *returnValue = errno;
   // return false;
}


const char* shellName()
{
   const char* shname = getenv("SHELL");
   if ( shname == 0 )
      shname = "/bin/sh";
   return shname;
}

const char* shellParam()
{
   return "-c";
}

bool openProcess(Process* _ph, String** arg_list, bool sinkin, bool sinkout, bool sinkerr, bool mergeErr, bool bg )
{
   PosixProcess* ph = static_cast<PosixProcess*>(_ph);

   // step 1: prepare the needed pipes
   if ( sinkin )
      ph->m_file_des_in[1] = -1;
   else
   {
      if ( pipe( ph->m_file_des_in ) < 0 )
      {
         ph->lastError(errno);
         return false;
      }
   }

   if ( sinkout )
      ph->m_file_des_out[0] = -1;
   else
   {
      if ( pipe( ph->m_file_des_out ) < 0 )
      {
         ph->lastError(errno);
         return false;
      }
   }

   if ( sinkerr )
      ph->m_file_des_err[0] = -1;
   else if ( mergeErr )
      ph->m_file_des_err[0] = ph->m_file_des_out[0];
   else
   {
      if ( pipe( ph->m_file_des_err ) < 0 )
      {
         ph->lastError(errno);
         return false;
      }
   }

   //Second step: fork
   ph->m_pid = fork();

   if ( ph->m_pid == 0 )
   {
      int hNull = 0;
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
      LocalizedArgv argv( arg_list );
      execvp( argv.p[0], argv.p ); // never returns.
      _exit( -1 );
   }
   else
      return true;
}

//====================================================================
// PosixProcess system area.

PosixProcess::PosixProcess():
   Process()
{}


PosixProcess::~PosixProcess()
{
   if ( ! done() )
   {
      close();
      terminate( true );
      wait( true );
   }
}

bool PosixProcess::wait( bool block )
{
   int pval,res;
   res = waitpid( m_pid, &pval, block ? 0 : WNOHANG );
   if( res == m_pid )
   {
      done(true);
      processValue( WEXITSTATUS(pval) );

      return true;
   }
   else if( res == 0 )
   {
      done(false);
      return true;
   }

   lastError( errno );
   return false;
}


bool PosixProcess::close()
{
   if ( m_file_des_err[1] != -1 )
      ::close(m_file_des_err[1]);
   if ( m_file_des_out[0] != -1 )
      ::close( m_file_des_out[0] );
   if ( m_file_des_err[0] != -1 )
      ::close( m_file_des_err[0] );
   return true;
}

bool PosixProcess::terminate( bool severe )
{
   int sig = severe ? SIGKILL : SIGTERM;
   if( kill( m_pid, sig ) == 0)
      return true;

   lastError( errno );
   return false;
}

::Falcon::Stream *PosixProcess::inputStream()
{
   if( m_file_des_in[1] == -1 || done() )
      return 0;

   UnixFileSysData *data = new UnixFileSysData( m_file_des_in[1], 0 );
   return new FileStream( data );
}

::Falcon::Stream *PosixProcess::outputStream()
{
   if( m_file_des_out[0] == -1 || done() )
      return 0;

   UnixFileSysData *data = new UnixFileSysData( m_file_des_out[0], 0 );
   return new FileStream( data );
}

::Falcon::Stream *PosixProcess::errorStream()
{
   if( m_file_des_err[0] != -1 || done() )
      return 0;

   UnixFileSysData *data = new UnixFileSysData( m_file_des_err[0], 0 );
   return new FileStream( data );

}

Process* Process::factory()
{
   return new PosixProcess();
}

}} // ns Falcon::Sys

/* end of process_sys_unix.cpp */
