/*
   FALCON - The Falcon Programming Language.
   FILE: coremodule.cpp

   Core module -- main file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Apr 2011 12:25:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/coremodule.cpp"

#include <falcon/cm/coremodule.h>

#include <falcon/cm/barrier.h>
#include <falcon/cm/compile.h>
#include <falcon/cm/debugger.h>
#include <falcon/cm/event.h>
#include <falcon/cm/fence.h>
#include <falcon/cm/iff.h>
#include <falcon/cm/include.h>
#include <falcon/cm/inspect.h>
#include <falcon/cm/print.h>
#include <falcon/cm/gc.h>
#include <falcon/cm/generator.h>
#include <falcon/cm/log.h>
#include <falcon/cm/textstream.h>
#include <falcon/cm/textwriter.h>
#include <falcon/cm/textreader.h>
#include <falcon/cm/datawriter.h>
#include <falcon/cm/datareader.h>
#include <falcon/cm/siter.h>
#include <falcon/cm/mutex.h>
#include <falcon/cm/parallel.h>
#include <falcon/cm/selector.h>
#include <falcon/cm/semaphore.h>
#include <falcon/cm/stdfunctions.h>
#include <falcon/cm/syncqueue.h>
#include <falcon/cm/tokenizer.h>
#include <falcon/cm/vmcontext.h>
#include <falcon/cm/waiter.h>
#include <falcon/cm/vmprocess.h>
#include <falcon/cm/multitokenizer.h>
#include <falcon/stdhandlers.h>

#include <falcon/engine.h>

namespace Falcon {

CoreModule::CoreModule():
   Module("core")
{
   static ClassStream* classStream = static_cast<ClassStream*>(
            Engine::handlers()->streamClass());
   
   Function* ffor = new Ext::Function_ffor;
   ffor->setEta(true);

   m_ctw = new Ext::ClassTextWriter( classStream );
   m_ctr = new Ext::ClassTextReader( classStream );

   *this
      // Standard functions
      << new Ext::Compile
      << new Ext::FuncPrintl
      << new Ext::FuncPrint
      << new Ext::Inspect
      << new Ext::Iff
      << new Ext::Function_rest
      //<< new Ext::Function_epoch
      << new Ext::Function_include
      << new Ext::Function_seconds
      << new Ext::Function_sleep
      << new Ext::Function_quit
      << new Ext::Function_advance
      << new Ext::Function_input
      << new Ext::Function_int
      << new Ext::Function_numeric
      << new Ext::Function_passvp
      << new Ext::Function_firstOf
      
      << new Ext::Function_map
      << new Ext::Function_filter
      << new Ext::Function_reduce
      << new Ext::Function_cascade
      << new Ext::Function_perform
      << new Ext::Function_ffirstOf
      << new Ext::Function_makeEnum

      << ffor

      // Standard classes
      << new Ext::ClassBarrier
      << new Ext::ClassEvent
      << new Ext::ClassFence
      << new Ext::ClassParallel
      << new Ext::ClassSIter
      << new Ext::ClassMutex
      << new Ext::ClassGenerator
      << new Ext::ClassTextStream( classStream )
      << m_ctw
      << m_ctr
      << new Ext::ClassDataWriter( classStream )
      << new Ext::ClassDataReader( classStream )
      << new Ext::ClassSelector
      << new Ext::ClassSemaphore
      << new Ext::ClassSyncQueue
      << new Ext::ClassVMContextBase
      << new Ext::ClassVMProcess
      << new Ext::ClassWaiter
      << new Ext::ClassLog
      << new Ext::ClassTokenizer
      << new Ext::ClassMultiTokenizer
      ;

   this->addObject( new Ext::ClassGC );
   this->addObject( new Ext::ClassVMContext );
   this->addObject( new Ext::ClassDebugger );

   // Add the args global
   this->globals().add("args", Item(), true);
}

}

/* end of coremodule.cpp */
