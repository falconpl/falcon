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
#include <falcon/string.h>
#include <falcon/trace.h>

using namespace Falcon;

IntMode::IntMode( FalconApp* owner ):
   m_owner( owner )
{}



void IntMode::run()
{
      // prepare to trace the GC.
#if FALCON_TRACE_GC
   Engine::instance()->collector()->trace( true );
#endif

   VMachine& vm = m_vm;
   
   vm.textOut()->write( "Welcome to Falcon.\n" );
   
   Module* core = new CoreModule;
   vm.modSpace()->addModule( core, true, false );
   IntCompiler intComp(&vm);

   String tgt;
   String prompt = ">>> ";
   
   while( true )
   {
      if( ! read_line( prompt, tgt ) )
      {
         break;
      }
      
      TRACE("GO -- Read: \"%s\"", tgt.c_ize() );

      // ignore empty lines.
      if( tgt.size() != 0 )
      {
         try
         {
            IntCompiler::compile_status status = intComp.compileNext(tgt + "\n");
            // is the compilation complete? -- display a result.
            switch( status )
            {
               // in this case, always display the value of a.
               case IntCompiler::eval_t:
                  vm.textOut()->write(vm.regA().describe()+"\n");
                  break;

               // in this case we want to ignore nil
               case IntCompiler::eval_direct_t:
                  if( ! vm.regA().isNil() )
                     vm.textOut()->write(vm.regA().describe()+"\n");
                  break;

               // we're waiting for more...
               case IntCompiler::incomplete_t: break;
               //... or we have nothing to do
               case IntCompiler::ok_t: break;
            }
         }
         catch( Error* e )
         {
            // display the error and continue
            if( e->errorCode() == e_compile )
            {
               // in case of a compilation, discard the encapsulator.
               class MyEnumerator: public Error::ErrorEnumerator {
               public:
                  MyEnumerator( TextWriter* wr ):
                     m_wr(wr)
                  {}

                  virtual bool operator()( const Error& e, bool  ){
                     m_wr->write(e.describe()+"\n");
                     return true;
                  }
               private:
                  TextWriter* m_wr;
               } rator(vm.textOut());
               
               e->enumerateErrors( rator );
            }
            else {
               vm.textOut()->write(e->describe()+"\n");
            }
            
            e->decref();
         }

         // resets the prompt
         prompt = intComp.isComplete() ? ">>> " : "... ";
      }
      // else, it's ok to leave the prompt as it is.
   }

#if FALCON_TRACE_GC
   vm.textOut()->write("\nGarbage data history:\n");
   Engine::instance()->collector()->dumpHistory( vm.textOut() );
#endif

}

/* end of int_mode.cpp */
