/*
 * fib.cpp
 *
 *  Created on: 15/gen/2011
 *      Author: gian
 */

#include <cstdio>
#include <iostream>

#include <falcon/vm.h>
#include <falcon/syntree.h>
#include <falcon/localsymbol.h>
#include <falcon/statement.h>
#include <falcon/synfunc.h>
#include <falcon/error.h>

#include <falcon/trace.h>
#include <falcon/application.h>

#include <falcon/psteps/stmtreturn.h>
#include <falcon/psteps/stmtif.h>
#include <falcon/psteps/exprcall.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/psteps/exprsym.h>
#include <falcon/psteps/exprcompare.h>
#include <falcon/psteps/exprmath.h>

using namespace Falcon;

class FibApp: public Falcon::Application
{

public:
void go( int fibSize )
{
// create a program:
   // function fib( n )
   //    if n < 2
   //       return n
   //    else
   //       return fib(n-1) + fib(n-2)
   //    end
   // end
   //
   // return fib(30)
   //

   SynFunc fib( "fib" );
   Symbol* count = fib.symbols().addLocal("n");
   fib.paramCount(1);

   SynTree* ifTrue = new SynTree();
   ifTrue->append(new StmtReturn( count->makeExpression() ) );

   SynTree* ifFalse = new SynTree();
   ifFalse->append( new StmtReturn( new ExprPlus(
         &(new ExprCall( new ExprValue(&fib) ))->addParam( new ExprMinus( count->makeExpression(), new ExprValue(1))),
         &(new ExprCall( new ExprValue(&fib) ))->addParam( new ExprMinus( count->makeExpression(), new ExprValue(2)))
         ))
   );

   fib.syntree().append(
         new Falcon::StmtIf(
               new ExprLT( count->makeExpression(), new ExprValue(2) ),
               ifTrue,
               ifFalse
         )
   );

   std::cout << fib.syntree().describe().c_ize() << std::endl;

   // and now the main function
   ExprCall* call_fib = new ExprCall( new ExprValue(&fib) );
   call_fib->addParam( new ExprValue(fibSize) );

   SynFunc fmain( "__main__" );
   fmain.syntree().append(
         new StmtReturn( call_fib )
   );

   // And now, run the code.
   Falcon::VMachine vm;
   vm.currentContext()->call(&fmain,0);
   try {
      vm.run();

      String res;
      vm.currentContext()->topData().describe( res );
      res.c_ize();
      std::cout << "Top: " << (char*)res.getRawStorage() << std::endl;
   }
   catch( Error* e )
   {
      std::cout << "Error: " << e->describe().c_ize() << std::endl;
      e->decref();
   }
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
}

};

// This is just a test.
int main( int argc, char* argv[] )
{
   std::cout << "Fib test!" << std::endl;

   if ( argc > 2 )
   {
      TRACE_ON();
   }
   
   FibApp app;
   app.go(argc > 1 ? atoi( argv[1] ) : 33 );

   return 0;
}
