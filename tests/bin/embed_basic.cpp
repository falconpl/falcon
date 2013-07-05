/*
   FALCON - The Falcon Programming Language.
   FILE: embed_basic.cpp

   Basic embedding test.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 03 Jul 2013 15:57:24 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falcon.h>

#include <iostream>

using namespace Falcon;


/** This example compiles and runs a script given as a single parameter in the command line.
 * If not given, the example will use the string
 * @code
 *    > "Hello world!"
 * @endcode
 *
 * as a template script.
 *
 * This example also uses the Log facility of the Falcon engine (which is optional).
 */

int main( int argc, char* argv[] )
{
   int exit_value = 0;

   // first of all, init the engine.
   Engine::init();

   // we'll use the standard Falcon logger (totally optional).
   Log* log = Engine::instance()->log();

   // Create the virtual machine -- that is used also for textual output.
   VMachine* vm = new VMachine;
   Process* process = vm->createProcess();

   const char* scriptData;
   if( argc < 2 )
   {
      scriptData =  "> \"Hello world!\"";
   }
   else {
      scriptData = argv[1];
   }

   // Create a simple script...
   StringStream* sinput = new StringStream( scriptData );
   // ... we can read from.
   TextReader* tr = new TextReader(sinput);
   IntCompiler ic;

   try {
      // let's create a module out of the input code.
      Module* mod = ic.compile(tr, String("eval://") + argv[0], "<evaluated>", false);

      // some compilation error?
      if( mod == 0 )
      {
         // let's make it easy: just handle it as any other error.
         throw ic.makeError();
      }

      // add the moduel to the modspace to allow it importing/exporting stuff.
      process->modSpace()->add(mod);

      // run main function
      Function* mainfunc = mod->getMainFunction();
      if( mainfunc != 0 )
      {
         log->log(Log::fac_app, Log::lvl_info, String("Launching the script") );

         // start the main function.
         process->start( mainfunc, 0, 0 );

         // wait for the process to terminate.
         process->wait();

         // NOTE: using Process::start/Process::wait will cause the process to be discarded.
         // After ::wait, the process is terminated and left in an invalid state.

         // a "return <number>" statement from script will give its value back here...
         // ForceInteger will be 0 if the script returns a string, a non-number or just don't return.
         exit_value = (int) process->result().forceInteger();

         // log the result
         log->log(Log::fac_app, Log::lvl_info, String("Script complete, exit value: ").N(exit_value) );

         // It's nice to know if the script returned something other than an integer.
         log->log(Log::fac_app, Log::lvl_detail, String("Exit value as item: ") + process->result().describe(3,128) );
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
   delete vm;

   // last thing to do before closing the program is shutting down the engine.
   Engine::shutdown();
   return exit_value;
}

