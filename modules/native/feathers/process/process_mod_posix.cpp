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
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "process.h"
#include "process_mod.h"
#include "process_fm.h"

#include <falcon/pipestreams.h>
#include <falcon/autocstring.h>
#include <falcon/filedata_posix.h>

#ifdef __gnu_linux__
#include <linux/unistd.h>
#endif

#include <string.h>

#include <vector>

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
               switch( chr )
               {
               case '\\': addParam(temp); state = state_escape; break;
               case '"': addParam(temp); state = state_doublequote; break;
               case '\'': addParam(temp); state = state_singlequote; break;
               case ' ': case '\t': case '\r': case '\n': addParam(temp); state= state_ws; break;
               default: temp.append(chr); break;
               }
            }
            break;

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

      if( !temp.empty() )
      {
         tempVector.push_back( temp );
      }

      uint32 size = tempVector.size();
      p = new char*[size + 1];
      p[size] = 0;

      for(uint32 i = 0; i < size; i++ )
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


void Process::sys_close()
{
  // nothing needed on POSIX -- all done by wait and closing streams.
}

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
      throw FALCON_SIGN_XERROR(::Falcon::Feathers::ProcessError,
               FALCON_PROCESS_ERROR_TERMINATE,
               .desc(FALCON_PROCESS_ERROR_TERMINATE_MSG)
               .sysError((unsigned int) errno ) );
   }

   return true;
}

int64 Process::pid() const
{
   return (int64) _p->m_pid;
}



void Process::sys_wait()
{
   int status = 0;
   while( ::waitpid(_p->m_pid, &status, 0 ) < 0 && errno == EINTR)
      /* try again */
      ;

   m_exitval = -1;
   if( WIFEXITED( status ) )
   {
      // force > 0.
      m_exitval = (int) static_cast<unsigned int>(WEXITSTATUS(status));
   }
}

void Process::sys_open( const String& cmd, int params )
{
   // step 1: prepare the needed pipes
   if ( pipe( _p->m_file_des_in ) < 0 )
   {
      throw FALCON_SIGN_XERROR( ::Falcon::Feathers::ProcessError,
               FALCON_PROCESS_ERROR_OPEN_PIPE,
               .desc( FALCON_PROCESS_ERROR_OPEN_PIPE_MSG )
               .extra("IN pipe")
               .sysError((unsigned int) errno) );
   }

   if ( pipe( _p->m_file_des_out ) < 0 )
   {
      throw FALCON_SIGN_XERROR( ::Falcon::Feathers::ProcessError,
                        FALCON_PROCESS_ERROR_OPEN_PIPE,
                        .desc( FALCON_PROCESS_ERROR_OPEN_PIPE_MSG )
                        .extra("OUT pipe")
                        .sysError((unsigned int) errno));
   }

   if ( (params & MERGE_AUX) != 0 )
   {
      _p->m_file_des_err[0] = _p->m_file_des_out[0];
      _p->m_file_des_err[1] = _p->m_file_des_out[1];
   }
   else
   {
      if ( pipe( _p->m_file_des_err ) < 0 )
      {
         throw FALCON_SIGN_XERROR( ::Falcon::Feathers::ProcessError,
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
      if( (params & MERGE_AUX) == 0 )
      {
         ::close( _p->m_file_des_err[0] );
      }

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


      // sink if explicitly requested, or if merged with output, but output is being sunk
      if( (params & SINK_AUX) != 0 ||
          ((params & (MERGE_AUX|SINK_OUTPUT)) == (MERGE_AUX|SINK_OUTPUT)) )
      {
         dup2( hNull, STDERR_FILENO );
      }
      else if( (params & MERGE_AUX) != 0 )
      {
         dup2( _p->m_file_des_out[1], STDOUT_FILENO );
      }
      else
      {
         dup2( _p->m_file_des_err[1], STDERR_FILENO );
      }

      // Launch the EXECVP procedure.
      if( (params & USE_SHELL) != 0 )
      {
         const char* argv[] = {0,0,0,0};
         AutoCString strCommand(cmd);
         argv[0] = shellName();
         argv[1] = "-c";
         argv[2] = strCommand.c_str();
         argv[3] = 0;

         ::execv( argv[0], (char* const*) argv ); // never returns.
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
      if( (params & MERGE_AUX) == 0 )
      {
         ::close( _p->m_file_des_err[1] );
      }

      // save the system-specific file streams, if not sunk
      if ( (params & SINK_INPUT) == 0 )
      {
         m_stdIn = new WritePipeStream( new Sys::FileData(_p->m_file_des_in[1]) );
      }
      else {
         ::close( _p->m_file_des_in[1] );
      }

      if ( (params & SINK_OUTPUT) == 0 )
      {
         m_stdOut = new ReadPipeStream( new Sys::FileData(_p->m_file_des_out[0]) );
      }
      else
      {
         ::close( _p->m_file_des_out[0] );
      }

      if ( (params & (SINK_AUX|MERGE_AUX) ) == 0 )
      {
         m_stdErr = new ReadPipeStream( new Sys::FileData(_p->m_file_des_err[0]) );
      }
      else
      {
         ::close( _p->m_file_des_err[0] );
      }
   }

}

//====================================================================
// Simple process manipulation functions

uint64 processId()
{
   return (uint64) getpid();
}

uint64 threadId()
{
#ifdef __gnu_linux__
   return (int64) syscall( __NR_gettid );

#else
   return 0;
#endif
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

   if ( m_sysdata == 0 )
   {
      throw FALCON_SIGN_XERROR( ::Falcon::Feathers::ProcessError, FALCON_PROCESS_ERROR_ERRLIST3,
               .desc(FALCON_PROCESS_ERROR_ERRLIST3_MSG )
               .sysError((uint32) errno));
   }

}

ProcessEnum::~ProcessEnum()
{
   this->close();
}

bool ProcessEnum::next()
{

   DIR* procdir = static_cast<DIR*>(m_sysdata);
   struct dirent* de;

   // `DIR' implements a stream and readdir moves it forward.
   while ( (de = ::readdir( procdir ) ) != 0 )
   {
      // skip non pid entries
      if ( de->d_name[0] >= '0' && de->d_name[0] <= '9' )
         break;
   }
   if ( !de ) return false; // EOF
   
   char statent[ 64 ];
   ::snprintf( statent, 64, "/proc/%s/stat", de->d_name );
   FILE* fp = fopen( statent, "r" );
   if ( !fp )
   {
      throw FALCON_SIGN_XERROR( ::Falcon::Feathers::ProcessError, FALCON_PROCESS_ERROR_ERRLIST4,
                           .desc(FALCON_PROCESS_ERROR_ERRLIST4_MSG )
                           );
   }
   
   int32 p_pid, p_ppid;
   char status;
   char szName[1024];
   if ( ::fscanf( fp, "%d %s %c %d", &p_pid, szName, &status, &p_ppid ) != 4 )
   {
      ::fclose( fp );
      throw FALCON_SIGN_XERROR( ::Falcon::Feathers::ProcessError, FALCON_PROCESS_ERROR_ERRLIST4,
                     .desc(FALCON_PROCESS_ERROR_ERRLIST4_MSG )
                     );
   }
   
   m_pid = (int64) p_pid;
   m_ppid = (int64) p_ppid;
   ::fclose(fp);

   if ( szName[0] == '(' )
   {
      szName[ strlen( szName ) -1] = '\0';
      m_name.bufferize( szName + 1 );
   }
   else
   {
      m_name.bufferize( szName );
   }

   // read also the command line, which may be missing.
   ::snprintf( statent, sizeof(statent), "/proc/%s/cmdline", de->d_name );
   fp = ::fopen( statent, "r" );
   if ( !fp || fscanf( fp, "%s", szName ) != 1 )
   {
      szName[0] = 0;
      return true;
   }

   ::fclose( fp );
   m_commandLine.bufferize( szName );

   return true;
}


void ProcessEnum::close()
{
   if ( m_sysdata != 0 )
   {
      if( ::closedir( static_cast<DIR*>(m_sysdata) ) != 0 )
      {
         throw FALCON_SIGN_XERROR( ::Falcon::Feathers::ProcessError, FALCON_PROCESS_ERROR_ERRLIST2,
                             .desc(FALCON_PROCESS_ERROR_ERRLIST2_MSG )
                             .sysError( (uint32) errno )
                             );
      }
      m_sysdata = 0;
   }
}

}} // ns Falcon::Mod

/* end of process_mod_posix.cpp */
