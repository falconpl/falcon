
/*
   FALCON - The Falcon Programming Language.
   FILE: testmode.cpp

   Falcon command line -- Test mode support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 29 Jan 2013 19:12:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/falcon.h>
#include <falcon/modloader.h>
#include <falcon/trace.h>

#include "testmode.h"

using namespace Falcon;

void TestMode::setup()
{
   // TODO: Fill the categories.
}

void TestMode::perform()
{
   Log* log = Engine::instance()->log();

   setup();

   try
   {
      Directory* dir = Engine::instance()->vfs().openDir( m_options.test_dir );

      String name;

      // first, we build a script map, so we can run tests in order.
      while(dir->read(name))
      {
         // Check format "NNN-NNN.fal"
         if( name.size() == 11 && name.subString(8,11) == '.fal')
         {
            for( int i = 0; i < 7; ++i )
            {
               char n = name.getCharAt(i);

               if( (i == 3 && n != '-') ||( i != 3 && ( n < '0' || n > '9' )))
               {
                  continue;
               }
            }
         }

         // ok, file in correct format.
         m_scripts[name] = ""; // no result for now.
         log->log( Log::fac_app, Log::lvl_detail, String( "Adding script: " ) + name );
      }

      if( m_scripts != 0 )
      {
         log->log( Log::fac_app, Log::lvl_info, String( "Starting to test " ).N(m_scripts.size()).A( " scripts") );
      }
      else {
         log->log( Log::fac_app, Log::lvl_warn, String( "No script in format NNN-NNN.fal found in target directory.") );
      }

      report();
   }
   catch(Error* e)
   {
      // let the main catcher to do the logging.
      throw e;
   }
}

void TestMode::report()
{
   TextStream ts( new StdOutStream );
   ts.write
}


void TestMode::test( ScriptData* sd )
{
}


/* end of testmode.cpp */

