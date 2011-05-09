/*
   FALCON - The Falcon Programming Language.
   FILE: intmode.cpp

   Falcon interactive compiler test.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 09 May 2011 19:04:08 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <cstdio>
#include <iostream>

#include <falcon/vm.h>
#include <falcon/syntree.h>
#include <falcon/localsymbol.h>
#include <falcon/error.h>
#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/exprsym.h>
#include <falcon/statement.h>
#include <falcon/stmtrule.h>
#include <falcon/rulesyntree.h>
#include <falcon/synfunc.h>
#include <falcon/extfunc.h>

#include <falcon/stdstreams.h>
#include <falcon/textwriter.h>

#include <falcon/trace.h>
#include <falcon/application.h>

#include <falcon/sourceparser.h>
#include <falcon/textreader.h>

#include <falcon/intcompiler.h>
#include <falcon/globalsymbol.h>
#include <falcon/genericerror.h>

using namespace Falcon;

//==============================================================
// The application
//

class ParserApp: public Falcon::Application
{

public:
   void guardAndGo()
   {
      try {
         go();
      }
      catch( Error* e )
      {
         std::cout << "Caught: " << e->describe().c_ize() << std::endl;
         e->decref();
      }
   }

void go()
{
   VMachine vm;
   IntCompiler intComp(&vm);

   String tgt;
   String prompt = ">>> ";
   
   while( ! vm.stdIn()->eof() )
   {
      vm.textOut()->write( prompt );
      vm.textOut()->flush();
      vm.textIn()->readLine(tgt, 4096);
      TRACE("GO -- Read: \"%s\"", tgt.c_ize() );

      // ignore empty lines.
      if( tgt.size() != 0 )
      {
         try
         {
            IntCompiler::compile_status status = intComp.compileNext(tgt + "\n");
            // is the compilation complete? -- display a result.
            if( status == IntCompiler::t_ok )
            {
               vm.textOut()->write(vm.regA().describe()+"\n");
               prompt = ">>> ";
            }
            else
            {
               // we're waiting for more
               prompt = "... ";
            }
         }
         catch( Error* e )
         {
            // display the error and continue
            vm.textOut()->write(e->describe()+"\n");
            e->decref();
         }
      }
      // else, it's ok to leave the prompt as it is.
   }
      
}

};

// This is just a test.
int main( int argc, char* argv[] )
{
   std::cout << "Interactive mode test" << std::endl;

   TRACE_ON();

   ParserApp app;
   app.guardAndGo();

   return 0;
}
