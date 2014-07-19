/*
   FALCON - The Falcon Programming Language.
   FILE: main.cpp

   Example of dynamic code compilation.

   This shows how to compile and execute dynamic code.
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


int main(int argc, char* argv[] )
{
   Falcon::Engine::init();
   Falcon::VMachine vm;

   std::cout << "The Falcon Programming Language" << std::endl
             << "Embedding test 003 -- Dynamic code" << std::endl
            ;

   if( argc < 2 )
   {
      std::cout << "Usage: 003 <code to be compiled>" << std::endl;
   }
   else
   {
      // create a process.
      Falcon::Process* myProcess = vm.createProcess();
      // add the core module
      myProcess->modSpace()->add( new Falcon::CoreModule );

      // let's try to run the script.
      try {
         myProcess->startScript(argv[1], false, "DynMod");
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
