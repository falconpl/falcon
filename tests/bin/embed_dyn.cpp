/*
   FALCON - The Falcon Programming Language.
   FILE: embed_sync.cpp

   Test for synchronous compilation and execution of dynamic code.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 03 Jul 2013 15:57:24 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falcon.h>
using namespace Falcon;

#include <iostream>

/** Test for synchronous compilation and execution of dynamic code.
 */

int main( int , char*[] )
{
   int exit_value = 0;

   // first of all, init the engine.
   Engine::init();

   // Create the virtual machine -- that is used also for textual output.
   VMachine* vm = new VMachine;
   Process* process = vm->createProcess();

   // and a context where we'll run the functions.
   WVMContext* wctx = new WVMContext( process );
   process->modSpace()->add(Engine::instance()->getCore());

   std::cout << "Expression to run (no multiline expression; quit exit)" << std::endl;
   vm->setProcessorCount(0);

   while( true )
   {
      char buffer[1024];
      std::cout << "INPUT> " << std::flush;
      std::cin.getline(buffer,1024);
      String script(buffer);
      if( script == "quit" ) { break; }

      std::cout << "Starting evaluation..." << std::endl;
      std::cout << "============================================================" << std::endl;
      try {
         wctx->startEvaluation( script+"\n" );
         wctx->wait();
         std::cout << "============================================================" << std::endl;
         vm->textOut()->writeLine( String("Evaluation result: ") + wctx->result().describe(3,-1) );
      }
      catch( Error* err ) {
         std::cout << "============================================================" << std::endl;
         std::cout << "The evaluation of this code generated an error:" << std::endl;
         vm->textOut()->writeLine( err->describe( true ) );
      }
   }

   // not really necessary here -- shutdown() will clear everything as needed.
   process->decref();
   wctx->decref();
   delete vm;

   // last thing to do before closing the program is shutting down the engine.
   Engine::shutdown();
   return exit_value;
}

