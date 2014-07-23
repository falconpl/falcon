/*
   FALCON - The Falcon Programming Language.
   FILE: main.cpp

   Advanced Event-driven Embedding (EDE) model sample.

   This test shows how to use the EventCourier class, which automates
   the task of providing an EDE model to scripts.

   EventCourier provides an high-performance, low-latency message
   transport which injects event and invokes callbacks in a target
   script-based application.

   The application needs not to be quiescent while receiving events
   from the EventCourier class (it can have contexts running in
   parallel with the context listening for events), and can use a
   script-reflected class to use the same message sending scheme and
   excite the same callbacks from within the script.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Jul 2014 18:19:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falcon.h>
#include <falcon/pstep.h>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>

#define EVENT_1_HANDLER "on_one"
#define EVENT_2_HANDLER "on_two"

#define EVENT_OBJECT_HANDLER "Handler"
#define EVENT_A_HANDLER "on_A"
#define EVENT_B_HANDLER "on_B"

#define EVENT_1_ID   0
#define EVENT_2_ID   1
#define EVENT_A_ID   2
#define EVENT_B_ID   3

int go(Falcon::EventCourier* evtc)
{
   std::cout << "Event handler list: "<< std::endl
            << EVENT_1_HANDLER << ": " << (evtc->hasCallback(EVENT_1_ID) ? "set" : "not set") << std::endl
            << EVENT_2_HANDLER << ": " << (evtc->hasCallback(EVENT_2_ID) ? "set" : "not set") << std::endl
            << EVENT_A_HANDLER << ": " << (evtc->hasCallback(EVENT_A_ID) ? "set" : "not set") << std::endl
            << EVENT_B_HANDLER << ": " << (evtc->hasCallback(EVENT_B_ID) ? "set" : "not set") << std::endl
            << "Default handler: " << (evtc->hasDefaultCallback() ? "set" : "not set") << std::endl
            ;

   std::cout << "Enter an event followed by parameters ('quit' to end): "<< std::endl;

   // let's try to run the script.
   try
   {

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

         // if we're here, we have a valid event.
         Falcon::Item params[64];
         for( size_t i = 1; i < command.size() && i < 64; ++i )
         {
            params[i-1] = FALCON_GC_HANDLE(new Falcon::String(command[i].c_str()));
         }

         Falcon::Event completed;
         Falcon::EventCourier::Token* token = 0;
         if(command[0] == "1")
         {
            token = evtc->sendEvent(EVENT_1_ID, params, command.size()-1, &completed );
         }
         else if(command[0] == "2" )
         {
            token = evtc->sendEvent(EVENT_2_ID, params, command.size()-1, &completed );
         }
         else if( (command[0] == "a" || command[0] == "A") )
         {
            token = evtc->sendEvent(EVENT_A_ID, params, command.size()-1, &completed );
         }
         else if( (command[0] == "b" || command[0] == "B") )
         {
            token = evtc->sendEvent(EVENT_B_ID, params, command.size()-1, &completed );
         }
         else if( command[0] == "quit" ) {
            break;
         }
         else {
            std::cout << "Unrecognized event '" << command[0] << "'" << std::endl;
            continue;
         }

         // Did we post some event?
         if( token != 0 )
         {
            try {
               Falcon::Item result;
               token->wait(result);

               Falcon::AutoCString cres( result.describe() );
               std::cout << ":: " << cres.c_str() << std::endl;
            }
            catch( Falcon::Error* e )
            {
               Falcon::AutoCString cres( e->describe() );
               e->decref();
               std::cout << "!! Event processing terminated with error: " << cres.c_str() << std::endl;
            }

            token->decref();
         }
      }

      // we're out of business.
      evtc->terminate();
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

   return 0;
}

//===============================================================================
// The application
//===============================================================================


int main(int argc, char* argv[] )
{
   Falcon::Engine::init();


   // This is our event courier; We use dynamic allocation to better control the deletion order.
   Falcon::VMachine* vm = new Falcon::VMachine;
   Falcon::EventCourier* courier = new Falcon::EventCourier;

   std::cout << "The Falcon Programming Language" << std::endl
             << "Embedding test 006 -- EventCourier high-performance Event handler." << std::endl
             << std::endl
             << "To generate an event, write 1, 2, a or b followed by a list of paramters." << std::endl
             ;

   if( argc < 2 )
   {
      std::cout << "Usage: 006 <scriptname>" << std::endl;
   }
   else
   {
      // create a process.
      Falcon::Process* myProcess = vm->createProcess();
      // and, why not, add the core module as well.
      myProcess->modSpace()->add( Falcon::Engine::instance()->getCore() );

      // get our own version of the courier handler class, so to be able to add our constants:
      Falcon::Class* ch = courier->handler()->createChild("ExtCourier");
      ch->addConstant("EVT_1", EVENT_1_ID);
      ch->addConstant("EVT_2", EVENT_2_ID);
      ch->addConstant("EVT_A", EVENT_A_ID);
      ch->addConstant("EVT_B", EVENT_B_ID);

      // we can export a symbol directly in the Module Space, without an application module.
      Falcon::Item icourier(ch, courier);

      // The script will see as "courier" as the global variable AppCourier, of class ExtCourier.
      myProcess->modSpace()->setExportValue("AppCourier", icourier);

      // Now the script has the occasion to configure itself through AppCourier
      try
      {
         myProcess->startScript(Falcon::URI(argv[1]), true);
         myProcess->wait();

         myProcess->reset();
         // The courier will now install itself below the exit point of the script
         courier->prepareContext(myProcess->mainContext());
         myProcess->start();

         // helper for the callbacks.
         go( courier );
         // on courier termination, the process will end.
         myProcess->wait();
         // we're done with the process.
         myProcess->decref();
      }
      catch( Falcon::Error* error )
      {
         Falcon::AutoCString desc( error->describe(true,true,false) );
         std::cout << "FATAL: Script terminated with error:" << std::endl;
         std::cout << desc.c_str() << std::endl;
      }
   }

   // here we can clear our event courier, before the engine shutdown.
   delete courier;
   delete vm;
   Falcon::Engine::shutdown();
   return 0;
}

