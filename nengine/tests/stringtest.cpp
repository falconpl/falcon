/*
 * stringtest.cpp
 *
 *  Created on: 15/gen/2011
 *      Author: gian
 */

#include <cstdio>
#include <iostream>

#include <falcon/vm.h>
#include <falcon/syntree.h>
#include <falcon/localsymbol.h>
#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/exprsym.h>
#include <falcon/statement.h>
#include <falcon/function.h>

#include <falcon/trace.h>
#include <falcon/application.h>

using namespace Falcon;

class StringApp: public Falcon::Application
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
   // create a program:
   // function string_add( param )
   //    return param + " -- Hello world"
   // end
   //
   // return string_add( "A string" )
   //

   Function string_add( "string_add" );
   Symbol* param = string_add.addVariable("param");
   string_add.paramCount(1);

   string_add.syntree().append(
         new StmtReturn( new ExprPlus( param->makeExpression(), new ExprValue( " -- Hello world" ) ) )
   );

   // and now the main function
   ExprCall* call_func = new ExprCall( new ExprValue(&string_add) );
   call_func->addParameter( new ExprValue("A string") );

   Function fmain( "__main__" );
   fmain.syntree().append(
         new StmtReturn( call_func )
   );

   // And now, run the code.
   Falcon::VMachine vm;
   vm.call(&fmain,0);
   vm.run();

   /*
   while( ! vm.codeEmpty() )
   {
      String report;
      vm.report( report );
      report.c_ize();
      std::cout << (char*)report.getRawStorage() << std::endl;
      vm.step();
   }
   */

   std::cout << "Top: " << vm.regA().describe().c_ize() << std::endl;
}

};

// This is just a test.
int main( int argc, char* argv[] )
{
   std::cout << "string test!" << std::endl;

   TRACE_ON();
   
   StringApp app;
   app.guardAndGo();

   return 0;
}
