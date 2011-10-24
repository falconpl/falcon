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
#include "falcon/modloader.h"
#include "falcon/modgroup.h"

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
         launch( script );
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


void FalconApp::launch( const String& script )
{
   // Create the virtual machine -- that is used also for textual output.
   VMachine vm;
   vm.setStdEncoding( m_options.io_encoding );
   
   // can we access the required module?
   Stream* fs = Engine::instance()->vfs().openRO( script );   
   if( fs == 0 )
   {
      vm.textOut()->write( "Can't open " + script+ "\n" );
      return;
   }
   
   // Ok, we opened the file; prepare the space (and most important, the loader) 
   ModSpace* ms = vm.modSpace();
   ModLoader* loader = ms->modLoader();
   
   // do we have a load path?
   loader->setSearchPath(".");
   if( m_options.load_path.size() > 0 )
   {
      // Is the load path totally substituting?
      loader->addSearchPath(m_options.load_path);
   }
   
   if( ! m_options.ignore_syspath )
   {
      loader->addFalconPath();
   }
   
   // Now configure other options of the lodaer.
   // -- How to treat sources?
   if( m_options.ignore_sources )
   {
      loader->useSources( ModLoader::e_us_never );
   }
   else if( m_options.force_recomp )
   {
      loader->useSources( ModLoader::e_us_always );
   }
   
   // -- Save modules?
   if( m_options.save_modules )
   {
      if( m_options.force_recomp)
      {
         loader->savePC( ModLoader::e_save_mandatory );
      }
      else
      {
         loader->savePC( ModLoader::e_save_try );
      }
   }
   else
   {
      loader->savePC( ModLoader::e_save_no );
   }   
   
   // What kind of module we are loading here?
   ModLoader::t_modtype type = ModLoader::e_mt_none;
   if( m_options.run_only )
   {
      type = ModLoader::e_mt_vmmod;
   }
   else if ( m_options.compile_only )
   {
      type = ModLoader::e_mt_source;
   }
         
   // TODO -- FIX ftds
   Module* module = loader->loadFile( script, type );
   module->setMain();
   
   Engine::instance()->getCore()->exportToModspace( vm.modSpace() );
   
   // and start the resolution dance.
   ModGroup* mg = new ModGroup( ms );  
   mg->add( module, e_lm_load );
   
   
   // throw on error.
   if( ! mg->link() )
   {
      throw mg->makeError();
   }
   
   mg->readyVM( vm.currentContext() );
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
