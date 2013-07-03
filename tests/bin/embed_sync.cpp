/*
   FALCON - The Falcon Programming Language.
   FILE: embed_sync.cpp

   Test for synchronous function invocation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 03 Jul 2013 15:57:24 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falcon.h>
using namespace Falcon;

#include <iostream>

/** Script executing an arbitrary function from a static module
 *  using the same context over and over again.
 */

int main( int argc, char* argv[] )
{
   int exit_value = 0;

   // first of all, init the engine.
   Engine::init();

   // Create the virtual machine -- that is used also for textual output.
   VMachine* vm = new VMachine;
   Process* process = vm->createProcess();

   // and a context where we'll run the functions.
   WVMContext* wctx = new WVMContext( process );

   if( argc < 2 )
   {
      std::cout << "Please, give a script name in the command line." << std::endl;
      return 1;
   }

   process->modSpace()->add(Engine::instance()->getCore());
   Process* loader = process->modSpace()->loadModule( argv[1], true, true, true );
   if( loader == 0 )
   {
      std::cout << "Can't find " << argv[1] << std::endl;
      return 1;
   }

   try {
      // Load the external script
      loader->start();
      loader->wait();

      // here is our module.
      Module* mod = static_cast<Module*>(loader->result().asInst());

      while( true )
      {
         std::string name;
         std::cout << "Item to run (quit to leave) > " << std::flush;
         std::cin >> name;
         if( name == "quit" ) { break; }

         Item* func = mod->globals().getValue( name.c_str() );
         if( func == 0 )
         {
            std::cout << "Not found \"" << name << "\"" << std::endl;
         }
         else {
            std::cout << "Starting evaluation of \"" << name << "\"..." << std::endl;
            std::cout << "============================================================" << std::endl;
            wctx->startItem(*func );
            wctx->wait();
            std::cout << "============================================================" << std::endl;
            std::cout << "Evaluation of \"" << name << "\" complete." << std::endl << std::endl;
         }
      }

      // not really necessary here -- shutdown() will clear everything as needed.
      mod->decref();
   }
   catch( Error* error )
   {
      std::cout << "Received an error: " << error->describe(false).c_ize() << std::endl;
      error->decref();
   }

   // not really necessary here -- shutdown() will clear everything as needed.
   process->decref();
   wctx->decref();
   delete vm;

   // last thing to do before closing the program is shutting down the engine.
   Engine::shutdown();
   return exit_value;
}

