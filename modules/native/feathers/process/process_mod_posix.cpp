/*
   FALCON - The Falcon Programming Language.
   FILE: process_mod_posix.cpp

   Unix specific implementation of openProcess
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Sep 2013 19:56:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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

#include "process.h"
#include "process_mod.h"
#include "process_ext.h"

#include <falcon/fstream.h>
#include <falcon/autocstring.h>
#include <falcon/filedata_posix.h>

#include <string.h>

namespace Falcon {
namespace Mod {

static const char* shellName()
{
   const char* shname = getenv("SHELL");
   if ( shname == 0 )
      shname = "/bin/sh";
   return shname;
}


namespace {

struct LocalizedArgv
{
   char** p;
   std::vector<String> tempVector;

   LocalizedArgv(const String& argList) :
      p( 0 )
   {
      this->fill( argList );
   }

    ~LocalizedArgv()
    {
       this->free();
    }

   void fill( const String& argList )
   {
      this->free();

      // break the string
      typedef enum {
         state_ws,
         state_word,
         state_singlequote,
         state_doublequote,
         state_escape,
         state_escapedbl,
      } t_state;

      t_state state = state_ws;
      length_t pos = 0;
      length_t end = argList.length();
      String temp;

      while( pos < end )
      {
         char_t chr = argList.getCharAt(pos++);
         switch(state)
         {
         case state_ws:
            switch( chr )
            {
            case '\\': state = state_escape; break;
            case '"': state = state_doublequote; break;
            case '\'': state = state_singlequote; break;
            case ' ': case '\t': case '\r': case '\n': break;
            default: temp.append(chr); state = state_word; break;
            }
            break;

         case state_word:
            {
            case '\\': addParam(temp); state = state_escape; break;
            case '"': addParam(temp); state = state_doublequote; break;
            case '\'': addParam(temp); state = state_singlequote; break;
            case ' ': case '\t': case '\r': case '\n': addParam(temp); state= state_ws; break;
            default: temp.append(chr); break;
            }

         case state_escape:
           temp.append(chr);
           state = state_word;
           break;

         case state_doublequote:
            switch(chr)
            {
            case '\\': addParam(temp); state = state_escapedbl; break;
            case '"': addParam(temp); state = state_ws; break;
            default: temp.append(chr); break;
            }
            break;

         case state_singlequote:
            switch(chr)
            {
            case '\'': addParam(temp); state = state_ws; break;
            default: temp.append(chr); break;
            }
            break;

         case state_escapedbl:
             temp.append(chr);
             state = state_doublequote;
             break;
         }
      }

      std::vector::size_type size = tempVector.size();
      p = new char*[size + 1];
      p[size] = 0;

      for(std::vector::size_type i = 0; i < size; i++ )
      {
         const String& arg = tempVector[i];
         size_t nBytes = arg.length() * 4+1;
         p[i] = new char[ nBytes ];
         arg.toCString(p[i], nBytes );
      }
   }

   void addParam( String& temp )
   {
      tempVector.push_back(temp);
      temp.clear();
   }

   void free()
   {
      tempVector.clear();
      if( !p ) return;

      for(size_t i = 0; p[i] != 0; i++ )
         delete [] p[i];
      delete [] p;
   }
};
} // end of anonymous namespace

class Process::Private
{
public:
   int m_file_des_in[2];
   int m_file_des_out[2];
   int m_file_des_err[2];

   pid_t m_pid;

   Private() {}
   ~Private() {}
};


void Process::sys_init()
{
   _p = new Private();
}

void Process::sys_destroy()
{
   delete _p;
}

bool Process::terminate( bool severe )
{
   int sig = severe ? SIGKILL : SIGTERM;
   m_mtx.lock();
   if( (! m_bOpen) || m_done )
   {
      m_mtx.unlock();
      return false;
   }
   m_mtx.unlock();

   if( ::kill( _p->m_pid, sig ) != 0 )
   {
      throw FALCON_SIGN_XERROR(::Falcon::Ext::ProcessError,
               FALCON_PROCESS_ERROR_TERMINATE,
               .desc(FALCON_PROCESS_ERROR_TERMINATE_MSG)
               .sysError((unsigned int) errno ) );
   }


   return false;
}



void Process::sys_open( const String& cmd, int params )
{
   // step 1: prepare the needed pipes
   if ( (params & SINK_INPUT) != 0 )
   {
      _p->m_file_des_in[1] = -1;
   }
   else
   {
      if ( pipe( _p->m_file_des_in ) < 0 )
      {
         throw FALCON_SIGN_XERROR( ::Falcon::Ext::ProcessError,
                  FALCON_PROCESS_ERROR_OPEN_PIPE,
                  .desc( FALCON_PROCESS_ERROR_OPEN_PIPE_MSG )
                  .extra("IN pipe")
                  .sysError((unsigned int) errno) );
      }
   }

   if ( (params & SINK_OUTPUT) != 0 )
   {
      _p->m_file_des_out[0] = -1;
   }
   else
   {
      if ( pipe( _p->m_file_des_out ) < 0 )
      {
         throw FALCON_SIGN_XERROR( ::Falcon::Ext::ProcessError,
                           FALCON_PROCESS_ERROR_OPEN_PIPE,
                           .desc( FALCON_PROCESS_ERROR_OPEN_PIPE_MSG )
                           .extra("OUT pipe")
                           .sysError((unsigned int) errno));
      }
   }


   if ( (params & SINK_AUX) != 0 )
   {
      _p->m_file_des_err[0] = -1;
   }
   else if ( (params & MERGE_AUX) != 0 )
   {
      _p->m_file_des_err[0] = _p->m_file_des_out[0];
   }
   else
   {
      if ( pipe( _p->m_file_des_err ) < 0 )
      {
         throw FALCON_SIGN_XERROR( ::Falcon::Ext::ProcessError,
               FALCON_PROCESS_ERROR_OPEN_PIPE,
               .desc( FALCON_PROCESS_ERROR_OPEN_PIPE_MSG )
               .extra("AUX pipe")
               .sysError((unsigned int) errno));
      }
   }

   //Second step: fork
   _p->m_pid = fork();

   // in the child?
   if ( _p->m_pid == 0 )
   {
      int hNull = 0;
      // close unused pipe ends
      ::close( _p->m_file_des_in[1] );
      ::close( _p->m_file_des_out[0] );
      ::close( _p->m_file_des_err[0] );

      // do we need to sink?
      if ( (params & (SINK_INPUT|SINK_OUTPUT|SINK_AUX)) != 0 )
      {
         hNull = ::open("/dev/null", O_RDWR);
      }

      // Third step: prepare the streams
      if ( (params & SINK_INPUT) != 0 )
      {
         dup2( hNull, STDIN_FILENO );
      }
      else
      {
         dup2( _p->m_file_des_in[0], STDIN_FILENO );
      }

      if ( (params & SINK_OUTPUT) != 0 )
      {
         dup2( hNull, STDOUT_FILENO);
      }
      else
      {
         dup2( _p->m_file_des_out[1], STDOUT_FILENO );
      }


      if( (params & SINK_AUX) != 0 )
      {
         dup2( hNull, STDERR_FILENO );
      }
      else if( (params & MERGE_AUX) != 0 )
      {
         dup2( _p->m_file_des_out[1], STDERR_FILENO );
      }
      else
      {
         dup2( _p->m_file_des_err[1], STDERR_FILENO );
      }

      // Launch the EXECVP procedure.
      if( (params & USE_SHELL) != 0 )
      {
         char* const argv[4];
         AutoCString strCommand(cmd);
         argv[0] = shellName();
         argv[1] = "-c";
         argv[2] = strCommand.c_str();
         argv[3] = 0;

         ::execv( argv[0], argv ); // never returns.
      }
      else
      {
         LocalizedArgv argv( cmd );
         if( (params & USE_PATH) != 0 )
         {
            ::execvp( argv.p[0], argv.p ); // never returns.
         }
         else
         {
            ::execv( argv.p[0], argv.p ); // never returns.
         }
      }

      ::_exit( -1 ); // just in case
   }
   else
   {
      // In the parent!

      // close unused pipe ends
      ::close( _p->m_file_des_in[0] );
      ::close( _p->m_file_des_out[1] );
      ::close( _p->m_file_des_err[1] );

      // save the system-specific file streams.
      m_stdIn = new WriteOnlyFStream( new Sys::FileData(_p->m_file_des_in[1]) );
      m_stdOut = new ReadOnlyFStream( new Sys::FileData(_p->m_file_des_out[0]) );
      m_stdErr = new ReadOnlyFStream( new Sys::FileData(_p->m_file_des_err[0]) );
   }

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
   this->close();
}

int ProcessEnum::next( String &name, uint64 &pid, uint64 &ppid, String &commandLine )
{
   if ( m_sysdata == 0 )
      return -1;

   DIR* procdir = static_cast<DIR*>(m_sysdata);
   struct dirent* de;

   // `DIR' implements a stream and readdir moves it forward.
   while ( (de = ::readdir( procdir ) ) != 0 )
   {
      // skip non pid entries
      if ( de->d_name[0] >= '0' && de->d_name[0] <= '9' )
         break;
   }
   if ( !de ) return 0; // EOF
   
   char statent[ 64 ];
   ::snprintf( statent, 64, "/proc/%s/stat", de->d_name );
   FILE* fp = fopen( statent, "r" );
   if ( !fp ) return -1;
   
   int32 p_pid, p_ppid;
   char status;
   char szName[1024];
   if ( ::fscanf( fp, "%d %s %c %d", &p_pid, szName, &status, &p_ppid ) != 4 )
   {
      ::fclose( fp );
      return -1;
   }
   
   pid = (int64) p_pid;
   ppid = (int64) p_ppid;
   ::fclose(fp);

   if ( szName[0] == '(' )
   {
      szName[ strlen( szName ) -1] = '\0';
      name.bufferize( szName + 1 );
   }
   else
      name.bufferize( szName );

   // read also the command line, which may be missing.
   ::snprintf( statent, sizeof(statent), "/proc/%s/cmdline", de->d_name );
   fp = ::fopen( statent, "r" );
   if ( !fp || fscanf( fp, "%s", szName ) != 1 )
   {
      szName[0] = 0;
      return 1;
   }

   ::fclose( fp );
   commandLine.bufferize( szName );

   return 1;
}

bool ProcessEnum::close()
{
   if ( m_sysdata != 0 )
   {
      closedir( static_cast<DIR*>(m_sysdata) );
      m_sysdata = 0;
      return true;
   }
   return false;
}

}} // ns Falcon::Mod

/* end of process_mod_posix.cpp */
