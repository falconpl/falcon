/*
   FALCON - The Falcon Programming Language.
   FILE: int_mode.cpp

   Falcon compiler and interpreter - interactive mode
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 23 Mar 2009 18:57:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "int_mode.h"
#include <falcon/modloader.h>
#include <falcon/module.h>
#include <falcon/synfunc.h>
#include <falcon/string.h>
#include <falcon/trace.h>
#include <falcon/psteps/pstep_compile.h>
#include <falcon/psteps/stmtreturn.h>

using namespace Falcon;

IntMode::IntMode( FalconApp* owner ):
   m_owner( owner )
{
   m_vm.modSpace()->add( Engine::instance()->getCore() );
}


void IntMode::run()
{
      // prepare to trace the GC.
#if FALCON_TRACE_GC
   Engine::instance()->collector()->trace( true );
#endif

   VMachine& vm = m_vm;
   
   vm.textOut()->write( "Welcome to Falcon.\n" );

   // add module and function
   Module *mod = new Module("(interactive)");
   SynFunc* mainfunc = new SynFunc("__main__");
   mod->setMain(true);
   mod->setMainFunction( mainfunc );
   vm.modSpace()->add(mod);


   
   // prepare the loader to fulfill dynamic load requests.

   // do we have a load path?
   ModLoader* loader = vm.modSpace()->modLoader();
   loader->setSearchPath(".");
   if( m_owner->m_options.load_path.size() > 0 )
   {
      // Is the load path totally substituting?
      loader->addSearchPath(m_owner->m_options.load_path);
   }
   
   if( ! m_owner->m_options.ignore_syspath )
   {
      loader->addFalconPath();
   }


   // Start the process.
   PStepCompile psc;
   vm.stdIn()->setNonblocking(true);
   psc.setCompilerContext(mainfunc, mod, vm.textIn(), vm.textOut() );
   mainfunc->syntree().append( new StmtReturn );
   Process* process = vm.createProcess();
   process->mainContext()->call( mainfunc );
   process->mainContext()->pushCode(&psc);

   process->start();
   process->wait();

#if FALCON_TRACE_GC
   vm.textOut()->write("\nGarbage data history:\n");
   Engine::instance()->collector()->dumpHistory( vm.textOut() );
#endif

}

/* end of int_mode.cpp */
