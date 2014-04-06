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

#include <falcon/stderrors.h>

using namespace Falcon;

IntMode::IntMode( FalconApp* owner ):
   m_owner( owner )
{
}


void IntMode::run()
{
      // prepare to trace the GC.
#if FALCON_TRACE_GC
   Engine::instance()->collector()->trace( true );
#endif

   VMachine& vm = m_vm;

   if( ! vm.setStdEncoding( m_owner->m_options.io_encoding) )
   {
      throw FALCON_SIGN_XERROR( EncodingError, e_unknown_encoding, .extra(m_owner->m_options.io_encoding));
   }

   vm.textOut()->write( "Welcome to Falcon.\n" );
   
   Process* process = vm.createProcess();
   process->modSpace()->add(Engine::instance()->getCore());

   // add module and function
   Module *mod = new Module("(interactive)");
   SynFunc* mainfunc = new SynFunc("__main__");
   mod->setMain(true);
   mod->setMainFunction( mainfunc );
   process->modSpace()->add(mod);
   
   process->setBreakCallback(&m_owner->m_dbg);
   // prepare the loader to fulfill dynamic load requests.

   // do we have a load path?
   ModLoader* loader = process->modSpace()->modLoader();
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

   psc.setCompilerContext(mainfunc, mod, vm.textIn(), vm.textOut() );
   mainfunc->syntree().append( new StmtReturn );

   process->mainContext()->call( mainfunc );
   process->mainContext()->pushCodeWithUnrollPoint(&psc);

   process->start();
   process->wait();

#if FALCON_TRACE_GC
   vm.textOut()->write("\nGarbage data history:\n");
   Engine::instance()->collector()->dumpHistory( vm.textOut() );
#endif

}

/* end of int_mode.cpp */
