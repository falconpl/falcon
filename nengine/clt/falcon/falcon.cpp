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

using namespace Falcon;

//==============================================================
// The application
//

void FalconApp::guardAndGo()
{
   TextWriter out(new StdOutStream);
   try {
      interactive();
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


int main( int , char* [] )
{
   TRACE_ON();

   FalconApp app;
   app.guardAndGo();

   return 0;
}

/* end of falcon.cpp */
