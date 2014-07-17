/*
   FALCON - The Falcon Programming Language.
   FILE: main.cpp

   Basic embedding test.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Jul 2014 18:19:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falcon.h>

#include <iostream>

int main(int argc, char* argv[] )
{
   Falcon::Engine::init();
   Falcon::VMachine vm;

   std::cout << "The Falcon Programming Language" << std::endl
             << "Embedding test 000 -- basic embedding" << std::endl
            ;

   if( argc < 2 )
   {
      std::cout << "Usage: 000 <scriptname> <arg1>...<argN>" << std::endl;
   }
   else
   {
      // create a process.
      Falcon::Process* myProcess = vm.createProcess();

      // Create an array containing the parameters for the script.
      Falcon::ItemArray* args = new Falcon::ItemArray;
      for( int i = 2; i < argc; ++i )
      {
         args->append( FALCON_GC_HANDLE(new Falcon::String(argv[i])) );
      }

      // export the symbol as "args"
      Falcon::Item i_args( FALCON_GC_HANDLE(args) );
      myProcess->modSpace()->setExportValue("args", i_args );

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
