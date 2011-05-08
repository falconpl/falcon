/*
 * fact.cpp
 *  adapted from fib.cpp
 *  Created on: 28/feb/2011
 *      Author: Paul 
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
#include <falcon/synfunc.h>

#include <falcon/trace.h>
#include <falcon/application.h>

using namespace Falcon;

class FactApp: public Falcon::Application
{

public:
void go( int factSize )
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

   SynFunc fact( "fact" );
   Symbol* count = fact.symbols().addLocal("n");
   fact.paramCount(1);

   SynTree* ifTrue = new SynTree();
   ifTrue->append(new StmtReturn(  new ExprValue(1) ) );

   SynTree* ifFalse = new SynTree();
   ifFalse->append( new StmtReturn( new ExprTimes(
         count->makeExpression(),
         &(new ExprCall( new ExprValue(&fact) ))->addParam( new ExprMinus( count->makeExpression(), new ExprValue(1)))
         ))
   );

   fact.syntree().append(
         new Falcon::StmtIf(
               new ExprLT( count->makeExpression(), new ExprValue(2) ),
               ifTrue,
               ifFalse
         )
   );

   std::cout << fact.syntree().describe().c_ize() << std::endl;

   // and now the main function
   ExprCall* call_fact = new ExprCall( new ExprValue(&fact) );
   call_fact->addParam( new ExprValue(factSize) );

   SynFunc fmain( "__main__" );
   fmain.syntree().append(
         new StmtReturn( call_fact )
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

   String res;
   vm.regA().describe( res );
   res.c_ize();
   std::cout << "Top: " << (char*)res.getRawStorage() << std::endl;
}

};

// This is just a test.
int main( int argc, char* argv[] )
{
   std::cout << "Fact test!" << std::endl;

   if ( argc > 2 )
   {
      TRACE_ON();
   }
   
   FactApp app;
   app.go(argc > 1 ? atoi( argv[1] ) : 33 );

   return 0;
}
