/*
   FALCON - The Falcon Programming Language.
   FILE: main.cpp

   Scripts pushing callbacks in the application.

   This test shows how an application might let a script to
   suggest what callbacks it will use to respond
   to events.

   Basically, this is done by exposing a registerCallback() function
   to have the script registering its callbacks.

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

#define EVENT_1_ID   0
#define EVENT_2_ID   1
#define EVENT_A_ID   2
#define EVENT_B_ID   3

/*
 * Use an anonymous namespace if you don't want to export the functions and
 * classes across C/C++ sources.
 */
namespace
{
   /* We'll use this class as a singleton, it won't handle actual objects.
    * So, we can cut a bit of corners.
    */
   class ClassEvents: public Falcon::Class
   {
   public:
      ClassEvents();
      virtual ~ClassEvents();

      virtual void dispose( void* instance ) const;
      virtual void* clone( void* instance ) const;
      virtual void* createInstance() const;

      /*
       * We might have this everywher, but... isn't this a nice place?
       */
      Falcon::GCLock* m_evt1_lock;
      Falcon::GCLock* m_evt2_lock;
      Falcon::GCLock* m_evtA_lock;
      Falcon::GCLock* m_evtB_lock;

      // Just to be sure.
      Falcon::Mutex m_mtx;
   };

   void ClassEvents::dispose( void* ) const
   {
      // nothing to do
   }

   void* ClassEvents::clone( void* ) const
   {
      // nothing to do
      return 0;
   }

   void* ClassEvents::createInstance() const
   {
      return 0;
   }


   /*
    * A function to let the script register a callback.
    * string, and replicates the string the given number of times.
    *
    * If the string is not given the value " " is assumed.
    */
   FALCON_DECLARE_FUNCTION(register, "evt:N,callback:X")
   FALCON_DEFINE_FUNCTION_P1(register)
   {
      Falcon::int64 number;
      Falcon::Item* cb;

      // the FALCON_[N]PCHECK_GET macro helps to check for mandatory  parameters
      if( ! FALCON_NPCHECK_GET(0, Ordinal, number)
          || (cb = ctx->param(1)) == 0 )
      {
         // the Function::paramError() method will generate a standardized error description
         // for incongruent paramters.
         throw paramError();
      }

      // get our class.
      ClassEvents* cls = static_cast<ClassEvents*>(methodOf());

      switch((int) number )
      {
      case EVENT_1_ID:
         cls->m_mtx.lock();
         if( cls->m_evt1_lock != 0) {cls->m_evt1_lock->dispose();};
         cls->m_evt1_lock = Falcon::Engine::instance()->collector()->lock(*cb);
         cls->m_mtx.unlock();
         break;

      case EVENT_2_ID:
         cls->m_mtx.lock();
         if( cls->m_evt2_lock != 0) {cls->m_evt2_lock->dispose();};
         cls->m_evt2_lock = Falcon::Engine::instance()->collector()->lock(*cb);
         cls->m_mtx.unlock();
         break;

      case EVENT_A_ID:
         cls->m_mtx.lock();
         if( cls->m_evtA_lock != 0) {cls->m_evtA_lock->dispose();};
         cls->m_evtA_lock = Falcon::Engine::instance()->collector()->lock(*cb);
         cls->m_mtx.unlock();
         break;

      case EVENT_B_ID:
         cls->m_mtx.lock();
         if( cls->m_evtB_lock != 0) {cls->m_evtB_lock->dispose();};
         cls->m_evtB_lock = Falcon::Engine::instance()->collector()->lock(*cb);
         cls->m_mtx.unlock();
         break;


      default:
         throw paramError("Unkown event ID");
         break;
      }

      // every function must return its frame, or the engine will stay in the frame.
      ctx->returnFrame();
   }


   ClassEvents::ClassEvents():
            Class("Events"),
            m_evt1_lock(0),
            m_evt2_lock(0),
            m_evtA_lock(0),
            m_evtB_lock(0)
   {
      // add a static method
      addMethod( new FALCON_FUNCTION_NAME(register),true);

      // Add some useful constant to this class
      addConstant("EVT_1", EVENT_1_ID);
      addConstant("EVT_2", EVENT_2_ID);
      addConstant("EVT_A", EVENT_A_ID);
      addConstant("EVT_B", EVENT_B_ID);
   }

   ClassEvents::~ClassEvents()
   {
      if( m_evt1_lock != 0 ) { m_evt1_lock->dispose(); }
      if( m_evt2_lock != 0 ) { m_evt2_lock->dispose(); }
      if( m_evtA_lock != 0 ) { m_evtA_lock->dispose(); }
      if( m_evtB_lock != 0 ) { m_evtB_lock->dispose(); }
   }

   //===============================================================================
   // Moduele
   //===============================================================================

   class AppModule: public Falcon::Module
   {
   public:
      AppModule();
      virtual ~AppModule();
      // we want a place where to store and get our class.
      ClassEvents* m_clsEvt;
   };

   AppModule::AppModule():
            Module("AppModule")
   {
      // this time we do not register
      m_clsEvt = new ClassEvents;

      *this
         << m_clsEvt
         ;
   }

   AppModule::~AppModule()
   {}
}





int go(Falcon::Process* myProcess, ClassEvents* evt)
{
   std::cout << "Event handler list: "<< std::endl
            << EVENT_1_HANDLER << ": " << (evt->m_evt1_lock == 0 ? " not set" : "set") << std::endl
            << EVENT_2_HANDLER << ": " << (evt->m_evt2_lock == 0 ? " not set" : "set") << std::endl
            << EVENT_A_HANDLER << ": " << (evt->m_evtA_lock == 0 ? " not set" : "set") << std::endl
            << EVENT_B_HANDLER << ": " << (evt->m_evtB_lock == 0 ? " not set" : "set") << std::endl
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
         if(command[0] == "1" && evt->m_evt1_lock != 0 )
         {
            // event 1
            // you can assign function to items.
            event = evt->m_evt1_lock->item();
         }
         else if(command[0] == "2" && evt->m_evt2_lock != 0 )
         {
            event = evt->m_evt2_lock->item();
         }
         else if( (command[0] == "a" || command[0] == "A") && evt->m_evtA_lock != 0 )
         {
            event = evt->m_evtA_lock->item();
         }
         else if( (command[0] == "b" || command[0] == "B") && evt->m_evtB_lock != 0 )
         {
            event = evt->m_evtB_lock->item();
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
   Falcon::VMachine vm;

   std::cout << "The Falcon Programming Language" << std::endl
             << "Embedding test 005 -- Script self-registered entry points." << std::endl
             << "This application provides the singleton Events with the method 'register'" << std::endl
             << "to let the scripts register their own code as event handlers." << std::endl
             << std::endl
             << "To generate an event, write 1, 2, a or b followed by a list of paramters." << std::endl

             ;

   if( argc < 2 )
   {
      std::cout << "Usage: 005 <scriptname>" << std::endl;
   }
   else
   {
      // create a process.
      Falcon::Process* myProcess = vm.createProcess();
      // and, why not, the core module as well.
      myProcess->modSpace()->add( Falcon::Engine::instance()->getCore() );

      // Add our applicaiton module
      // we want to have a reference to the ClassEvent to use it in our application.
      AppModule* appmod = new AppModule;
      myProcess->modSpace()->add( appmod );

      // let's try to run the script.
      try {
         myProcess->startScript(Falcon::URI(argv[1]), true);
         // wait for the process to complete.
         myProcess->wait();

         // and now, we can go

         Falcon::AutoCString resultDesc( myProcess->result().describe(3, 128) );
         std::cout << "====================================================" << std::endl;
         std::cout << "Script completed with result: " << resultDesc.c_str() << std::endl;
         std::cout << "Now starting the event callbacks" << std::endl;
         std::cout << "====================================================" << std::endl;

         Falcon::Process* cbProc = vm.createProcess();
         // not completely necessary, but we might want give our scripts their own modules.
         cbProc->adoptModSpace( myProcess->modSpace() );
         return go( cbProc , appmod->m_clsEvt );
      }
      catch( Falcon::Error* error )
      {
         Falcon::AutoCString desc( error->describe(true,true,false) );
         std::cout << "FATAL: Script terminated with error:" << std::endl;
         std::cout << desc.c_str() << std::endl;
      }
   }

   Falcon::Engine::shutdown();
   return 0;
}

