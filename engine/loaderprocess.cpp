/*
   FALCON - The Falcon Programming Language.
   FILE: loaderprocess.cpp

   VM Process specialized in loading modules into the VM.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 12 Dec 2012 19:52:50 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/loaderprocess.h>
#include <falcon/vmcontext.h>
#include <falcon/module.h>
#include <falcon/engine.h>
#include <falcon/synfunc.h>
#include <falcon/modspace.h>

namespace Falcon {

LoaderProcess::LoaderProcess( VMachine* vm, ModSpace* ms ):
   Process( vm ),
   m_ms(ms),
   m_mainModule(0),
   m_stepSetupMain(this)
{
   m_loaderEntry = new SynFunc("#loaderEntry");
}

LoaderProcess::~LoaderProcess()
{
   if( m_mainModule != 0 ) {
      m_mainModule->decref();
   }
   delete m_loaderEntry;
}


void LoaderProcess::loadModule( const String& modUri, bool isUri, bool launchMain )
{
   m_context->reset();
   m_context->call( m_loaderEntry );
   m_context->pushCode( &m_stepSetupMain );

   m_ms->loadModuleInContext( modUri, isUri, false, launchMain, mainContext(), 0, true );
   start();
}


void LoaderProcess::setMainModule( Module* mod )
{
   if( m_mainModule != 0 ) {
      m_mainModule->decref();
   }

   if( mod != 0 ) {
      mod->incref();
   }

   m_mainModule = mod;
}

Module* LoaderProcess::mainModule() const
{
   m_mainModule->incref();
   return m_mainModule;
}


void LoaderProcess::PStepSetupMain::apply_( const PStep* self, VMContext* ctx )
{
   const LoaderProcess::PStepSetupMain* pstep = static_cast<const LoaderProcess::PStepSetupMain* >(self);

   // the module we're working on is at top data stack.
   Module* mod = static_cast<Module*>(ctx->topData().asInst());

   // mark this as the main module and proceed.
   pstep->m_owner->setMainModule( mod );
   mod->setMain(true);
   ctx->popData();
   ctx->popCode();
}

}

/* end of loaderprocess.cpp */
