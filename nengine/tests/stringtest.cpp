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
#include <falcon/error.h>
#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/exprsym.h>
#include <falcon/exprcall.h>
#include <falcon/exprmath.h>
#include <falcon/statement.h>
#include <falcon/stmtrule.h>
#include <falcon/rulesyntree.h>
#include <falcon/synfunc.h>
#include <falcon/extfunc.h>

#include <falcon/trace.h>
#include <falcon/application.h>

#include <falcon/cm/coremodule.h>
#include <falcon/globalsymbol.h>

using namespace Falcon;

class StringApp: public Falcon::Application
{
public:
   CoreModule cm;
   
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
   // str = string_add( "A string" )
   // printl( str )

   SynFunc string_add( "string_add" );
   Symbol* param = string_add.symbols().addLocal("param");
   string_add.paramCount(1);

   string_add.syntree().append(
         new StmtReturn( new ExprPlus( param->makeExpression(), new ExprValue( " -- Hello world" ) ) )
   );

   // and now the main function
   ExprCall* call_func = new ExprCall( new ExprValue(&string_add) );
   call_func->addParam( new ExprValue("A string") );

   Symbol* printl = cm.getGlobal("printl");

   // and the main
   SynFunc fmain( "__main__" );
   Falcon::Symbol* strsym = fmain.symbols().addLocal("str");

   fmain.syntree()
      .append( new StmtAutoexpr(
               new ExprAssign( strsym->makeExpression(), call_func ) ))
      .append( new StmtAutoexpr(&(new ExprCall( printl->makeExpression() ))
            ->addParam(strsym->makeExpression()).addParam(new ExprValue(1))) )
      .append( new StmtReturn( strsym->makeExpression() ));

   std::cout << "Will run: " << fmain.syntree().describe().c_ize() << std::endl;

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
int main( int , char* [] )
{
   std::cout << "string test!" << std::endl;

   TRACE_ON();
   
   StringApp app;
   app.guardAndGo();

   return 0;
}
