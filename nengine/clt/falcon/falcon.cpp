/*
   FALCON - The Falcon Programming Language.
   FILE: falcon.cpp

   Falcon command line
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 12 Jan 2011 20:37:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/falcon.h>
#include <falcon/trace.h>

#include "int_mode.h"
#include "falcon/modcompiler.h"
#include "falcon/streambuffer.h"

using namespace Falcon;

//==============================================================
// The application
//
FalconApp::FalconApp():
   m_exitValue(0)
{}

void FalconApp::guardAndGo( int argc, char* argv[] )
{
   int scriptPos = 0;
   m_options.parse( argc, argv, scriptPos );
   if( m_options.m_justinfo )
   {
      return;
   }
   
   TextWriter out(new StdOutStream);
   try 
   {
      if( m_options.interactive )
      {
         interactive();
      }
      else
      {
         if ( scriptPos <= 0 )
         {
            out.write( "Please, add a filename (for now)\n" );
            return;
         }
         
         String script = argv[scriptPos-1];
         compile( script );
      }
   }
   catch( Error* e )
   {
      out.write( "Caught: " + e->describe() +"\n");
      e->decref();
   }
}


void FalconApp::interactive()
{
   IntMode intmode( this );
   intmode.run();
}

void FalconApp::compile( const String& script )
{
   TextWriter out(new StdOutStream);
   
   Stream* fs = Engine::instance()->vsf().open( script, 
      VFSProvider::OParams().rdOnly() );
   
   if( fs == 0 )
   {
      out.write( "Can't open " + script+ "\n" );
      return;
   }
   
   ModCompiler compiler;
   
   TextReader* tr = new TextReader( fs );
   
   Module* mod = compiler.compile( tr, script, script );
   if( mod == 0 )
   {
      
      Error* err = compiler.makeError();         
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
      };
      MyEnumerator rator(&out);

      err->enumerateErrors( rator );
      return;
   }
   
   SynFunc* fmain = (SynFunc*) mod->getFunction("__main__"); 
   if( m_options.tree_out )
   {
      
      out.write( fmain->syntree().describe() +"\n");
      return;
   }
   
   
   VMachine vm;
   vm.link( mod );
   vm.currentContext()->call( fmain, 0 );
   vm.run();
}


int main( int argc, char* argv[] )
{
   TRACE_ON();

   FalconApp app;
   app.guardAndGo( argc, argv );
   
   return app.m_exitValue;
}

/* end of falcon.cpp */
