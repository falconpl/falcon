/*
   FALCON - The Falcon Programming Language.
   FILE: main.cpp

   Application pulling entry points from scripts.

   This test shows how an applicaiton can retreive callbacks and entry
   points from determined names in scripts, and then repeatedly calling
   them to respond to application events.

   This example uses a simple callback model based on the WVMContext,
   or Waitable Virtual Machine Context, that is a context that the
   application can wait on for termination.

   A more advanced model is available for high performance embedding
   (shown in the more advanced example).

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Jul 2014 18:19:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falcon.h>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>

#define EVENT_1_HANDLER "on_one"
#define EVENT_2_HANDLER "on_two"

#define EVENT_OBJECT_HANDLER "Handler"
#define EVENT_A_HANDLER "on_A"
#define EVENT_B_HANDLER "on_B"

int go(int argc, char* argv[] )
{

   Falcon::VMachine vm;

   std::cout << "The Falcon Programming Language" << std::endl
             << "Embedding test 004 -- Script provided entry points." << std::endl
             << EVENT_1_HANDLER << ": handler of the '1' event." << std::endl
             << EVENT_2_HANDLER << ": handler of the '2' event." << std::endl
             << EVENT_OBJECT_HANDLER << ": An object containing the following callbacks:" << std::endl
             << "   " << EVENT_A_HANDLER << ": handler responding to 'a' event." << std::endl
             << "   " << EVENT_B_HANDLER << ": handler responding to 'b' event." << std::endl
             << std::endl
             << "To generate an event, write 1, 2, a or b followed by a list of paramters." << std::endl
             ;

   if( argc < 2 )
   {
      std::cout << "Usage: 004 script" << std::endl;
   }
   else
   {
      Falcon::Module* mod = 0;

      // create a process.
      Falcon::Process* myProcess = vm.createProcess();
      // add the core module
      myProcess->modSpace()->add( new Falcon::CoreModule );

      // first of all, load the module -- without running it.
      // to to that, we must create a VM loader process, that will be separated from our
      // runner process.
      try {
         Falcon::Process* loader = myProcess->modSpace()->loadModule(argv[1], true, true, true );

         // wait for the loader to complete.
         loader->start();
         loader->wait();
         // recieved main module.
         mod = static_cast<Falcon::Module*>(loader->result().asInst());
         loader->decref();
      }
      catch(Falcon::Error* error)
      {
         Falcon::AutoCString desc( error->describe(true,true,false) );
         std::cout << "FATAL: Failed to load script:" << std::endl;
         std::cout << desc.c_str() << std::endl;
         Falcon::Engine::shutdown();
         return 1;
      }

      // save the callback items.

      // we explicitly want a function for this...
      Falcon::Function* i_one = mod->getFunction(EVENT_1_HANDLER);
      Falcon::Function* i_two = mod->getFunction(EVENT_2_HANDLER);
      // as these are functions stored in the module,
      // they can't be deleted 'til the module is alive, and we have
      // an extra reference to it.

      // while we want an object for this...
      // ... which is a "global variable" in the module
      Falcon::Item* i_obj = mod->resolveLocally(EVENT_OBJECT_HANDLER);
      Falcon::GCLock* i_obj_lock = 0;

      Falcon::Item i_a;
      Falcon::Item i_b;
      if( i_obj != 0 )
      {
         // tell the Falcon GC that we want this object to stay valid even if
         // the module, for any reason, loses track of it.
         i_obj_lock = Falcon::Engine::instance()->collector()->lock(*i_obj);
         Falcon::Class* cls = 0;
         void* inst = 0;
         i_obj->asClassInst(cls, inst);

         cls->getProperty( EVENT_A_HANDLER, inst, i_a );
         cls->getProperty( EVENT_B_HANDLER, inst, i_b );
      }

      std::cout << "Event handler list: "<< std::endl
               << EVENT_1_HANDLER << ": " << (i_one == 0 ? " not set" : "set") << std::endl
               << EVENT_2_HANDLER << ": " << (i_two == 0 ? " not set" : "set") << std::endl
               << EVENT_A_HANDLER << ": " << (i_a.isNil() ? " not set" : "set") << std::endl
               << EVENT_B_HANDLER << ": " << (i_b.isNil() ? " not set" : "set") << std::endl
               ;

      std::cout << "Enter an event followed by parameters ('quit' to end): "<< std::endl;

      // let's try to run the script.
      try
      {
         Falcon::WVMContext* ctx = new Falcon::WVMContext(myProcess);

         // repeat...
         while( true )
         {
            // get the command
            std::cout << ":> " << std::flush;
            std::string line;
            std::vector<std::string> command;

            std::getline(std::cin, line);
            std::istringstream iss(line);
            for (std::string word; iss >> word; ) { command.push_back(word); }

            if( command.size() == 0 )
            {
               continue;
            }

            Falcon::Item event;
            if(command[0] == "1" && i_one )
            {
               // event 1
               // you can assign function to items.
               event = i_one;
            }
            else if(command[0] == "2" && i_two )
            {
               event = i_two;
            }
            else if( command[0] == "a" || command[0] == "A" )
            {
               event = i_a;
            }
            else if( command[0] == "b" || command[0] == "B" )
            {
               event = i_b;
            }
            else if( command[0] == "quit" ) {
               break;
            }
            else {
               std::cout << "Unrecognized event '" << command[0] << "'" << std::endl;
               continue;
            }

            // we're creating stuff that will go in GC, but the context isn't running...
            ctx->registerInGC();

            // if we're here, we have a valid event.
            Falcon::Item params[64];
            for( size_t i = 1; i < command.size() && i < 64; ++i )
            {
               params[i-1] = FALCON_GC_HANDLE(new Falcon::String(command[i].c_str()));
            }

            ctx->startItem(event,command.size()-1,params);

            // now we could go doing other stuff, but...
            ctx->wait();

            Falcon::AutoCString cres( ctx->result().describe() );
            std::cout << ":: " << cres.c_str() << std::endl;

            // clear for the next time around.
            ctx->reset();
         }

         // clear the lock, if necessary
         if( i_obj_lock != 0 )
         {
            i_obj_lock->dispose();
         }
      }
      catch( Falcon::Error* error )
      {
         Falcon::AutoCString desc( error->describe(true,true,false) );
         std::cout << "FATAL: Script terminated with error:" << std::endl;
         std::cout << desc.c_str() << std::endl;
         return 1;
      }
      catch(std::exception& e )
      {
         std::cout << "FATAL: I/O Exception: " << e.what() << std::endl;
         return 1;
      }
   }

   return 0;
}

int main(int argc, char* argv[] )
{
   // now we screen the engine, as we'll be implicitly killing the VM
   Falcon::Engine::init();
   int value = go( argc, argv );
   Falcon::Engine::shutdown();


   return value;
}
