/*
   FALCON - The Falcon Programming Language.
   FILE: main.cpp

   Example of application provided functions, classes and modules.

   This progam demonstrates how the script can pull entities
   from the host application, via a dynamic name resolution
   function that is hooked in the applicaiton module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Jul 2014 18:19:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falcon.h>
#include <iostream>
#include <map>

/*
 * Use an anonymous namespace if you don't want to export the functions and
 * classes across C/C++ sources.
 */
namespace {

   /*
    * This defines a "function family" that just returns its own name,
    * like in the following example:
    *
    * > myfunc()  // "I am myfunc"
    *
    */
   class MyFunc: public Falcon::Function
   {
   public:
      MyFunc( const Falcon::String& name ):
         Function(name)
      {}

      virtual ~MyFunc() {}

      virtual void invoke( Falcon::VMContext* ctx, Falcon::int32 )
      {
         Falcon::String* retval = new Falcon::String("I am ");
         retval->append( this->name() );
         ctx->returnFrame( FALCON_GC_HANDLE(retval) );
      }
   };

   //===============================================================================
   // Moduele
   //===============================================================================

   class AppModule: public Falcon::Module
   {
   public:
      AppModule();
      virtual ~AppModule();
      // override this to provide entities on request:
      virtual Falcon::Item* resolveLocally( const Falcon::Symbol* sym );

   private:
      /*
       * Althoguh the Module has a dictionary of symbols we could use,
       * we create our own here to show the barebones of the process.
       */
      typedef std::map<Falcon::String, Falcon::Item> SymbolMap;
      SymbolMap mysyms;
   };

   AppModule::AppModule():
            Module("AppModule")
   {
      // we don't add anything...
   }

   AppModule::~AppModule()
   {
      // we own our functions, and when we're done, then it means our functions are clear to go too.
      SymbolMap::iterator iter = mysyms.begin();
      while( iter != mysyms.end() )
      {
         Falcon::Function* f = iter->second.asFunction();
         delete f;
         ++iter;
      }
   }

   Falcon::Item* AppModule::resolveLocally( const Falcon::Symbol* sym )
   {
      const Falcon::String& name = sym->name();

      Falcon::Item& value = mysyms[name];
      // still not initialized?
      if( value.isNil() )
      {
         Falcon::Function* f = new MyFunc(name);
         f->module(this);
         value = f;
      }

      return &value;
   }
}


//===============================================================================
// The application
//===============================================================================


int main(int argc, char* argv[] )
{
   Falcon::Engine::init();
   Falcon::VMachine vm;

   std::cout << "The Falcon Programming Language" << std::endl
             << "Embedding test 002 -- Advanced embedding" << std::endl
            ;

   if( argc < 2 )
   {
      std::cout << "Usage: 002 <scriptname> <arg1>...<argN>" << std::endl;
   }
   else
   {
      // create a process.
      Falcon::Process* myProcess = vm.createProcess();

      // Add our applicaiton module; it's wise to create a new module for each modspace,
      // although, if wished, and properly protected against concurrency,
      // you might share the same module across different ones.
      myProcess->modSpace()->add( new AppModule );

      // let's try to run the script.
      try {
         myProcess->startScript(Falcon::URI(argv[1]), true);
         // wait for the process to complete.
         myProcess->wait();

         Falcon::AutoCString resultDesc( myProcess->result().describe(3, 128) );
         std::cout << "====================================================" << std::endl;
         std::cout << "Script completed with result: " << resultDesc.c_str() << std::endl;
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
