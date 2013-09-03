/*
   FALCON - The Falcon Programming Language.
   FILE: process_mod.h

   System dependent module specifications for process module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Sep 2013 19:10:59 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/native/feathers/process/process_mod.h"

#include "process.h"
#include "process_mod.h"
#include <falcon/string.h>
#include <falcon/mt.h>
#include <falcon/stream.h>
#include <falcon/vmcontext.h>
#include <falcon/vm.h>

#include <falcon/stderrors.h>


namespace Falcon {
namespace Mod {

Process::Process( VMContext* ctx, const Class* handler ):
    Shared(&ctx->vm()->contextManager(), handler, false, 0),
    m_bOpen( false ),
    m_done(false),
    m_exitval( 0 ),
    m_twThread( 0 ),
    m_stdIn(0),
    m_stdOut(0),
    m_stdErr(0),
    m_tw(this)
{
   // system specific process class initialization.
   sys_init();
}

Process::~Process()
{
   if( m_stdIn != 0 ) { m_stdIn->decref(); }
   if( m_stdOut != 0 ) { m_stdOut->decref(); }
   if( m_stdErr != 0 ) { m_stdErr->decref(); }

   // system specfic process class destruction
   sys_destroy();
}


void Process::open( const String& args, int mode, bool async )
{
   m_mtx.lock();
   if( m_bOpen )
   {
      m_mtx.unlock();
      throw FALCON_SIGN_XERROR(CodeError,
               FALCON_PROCESS_ERROR_ALREADY_OPEN,
               .desc(FALCON_PROCESS_ERROR_ALREADY_OPEN_MSG));
   }
   else {
      m_bOpen = true;
      m_mtx.unlock();
   }

   // if sysopen fails, we throw
   try
   {
      // this open the resources and fills the std streams.
      sys_open( args, mode );
   }
   catch( ... )
   {
      m_mtx.lock();
      m_bOpen = false;
      m_mtx.unlock();
      throw;
   }

   m_cmd = args;
   if( async )
   {
      m_twThread = new SysThread( &m_tw );
      incref();
      // the thread will take care of itself -- start detached.
      if ( ! m_twThread->start(ThreadParams().stackSize(1024*20).detached(true)) )
      {
         throw FALCON_SIGN_XERROR(CodeError, e_binstartup, .extra("Thread startup") );
      }
   }
}

int32 Process::consumeSignal( VMContext*, int32 )
{
   int32 count = signalCount() > 0 ? 1 : 0;
   return count;
}


int Process::lockedConsumeSignal( VMContext*, int )
{
   int32 count = lockedSignalCount() > 0 ? 1 : 0;
   return count;
}


Stream* Process::inputStream() const
{
   return m_stdIn;
}

Stream* Process::outputStream() const
{
   return m_stdOut;
}

Stream* Process::errorStream() const
{
   return m_stdErr;
}


bool Process::exitValue(int &value) const
{
   m_mtx.lock();
   bool term = m_done;
   value = m_exitval;
   m_mtx.unlock();

   return term;
}


void Process::waitTermination()
{
   sys_wait();

   m_mtx.lock();
   m_done = true;
   m_mtx.unlock();
}

void Process::close()
{
   // close the streams, but don't decref them
   if(m_stdIn != 0 ) { m_stdIn->close(); }
   if(m_stdOut != 0 ) { m_stdOut->close(); }
   if(m_stdErr != 0 ) { m_stdErr->close(); }

   sys_close();
}


//====================================================================
// Class waiting for child process termination
//====================================================================

Process::TermWaiter::TermWaiter( Process* owner ):
         m_process(owner)
{
}

Process::TermWaiter::~TermWaiter()
{
}

void* Process::TermWaiter::run()
{
   m_process->waitTermination();

   // free the resources (in case it's needed).
   // of course -- don't close the streams, we need them
   m_process->sys_close();

   m_process->signal(1);

   // if we started async, we were given an extra ref.
   m_process->decref();
   return 0;
}

}} // NS Falcon::Mod

/* end of process_mod.cpp */
