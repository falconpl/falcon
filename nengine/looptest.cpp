/*
 * main.cpp
 *
 *  Created on: 12/gen/2011
 *      Author: gian
 */

#include <cstdio>
#include <iostream>

#include <falcon/vm.h>
#include <falcon/syntree.h>
#include <falcon/localsymbol.h>
#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/statement.h>

// This is just a test.
int main( int argc, char* argv[] )
{
   std::cout << "Hello world" << std::endl;

   // create a program:
   // count = 0
   // while count < 5
   //   count = count + 1
   // end

   Falcon::Symbol* count = new Falcon::LocalSymbol("count",0);
   Falcon::SynTree* assign = new Falcon::SynTree;
   assign->append(
         new Falcon::StmtAutoexpr(
               new Falcon::ExprAssign( count->makeExpression(),
                     new Falcon::ExprPlus( count->makeExpression(), new Falcon::ExprValue(1) )
         )));


   Falcon::SynTree* program = new Falcon::SynTree;
   (*program)
      .append( new Falcon::StmtAutoexpr(new Falcon::ExprAssign( count->makeExpression(), new Falcon::ExprValue(0) ) ) )
      .append( new Falcon::StmtWhile(
                     new Falcon::ExprLT( count->makeExpression(), new Falcon::ExprValue(50000000) ),
                     assign ) );


   Falcon::String res = program->toString();
   res.c_ize();
   std::cout << (char*)res.getRawStorage() << std::endl;

   // And now, run the code.
   Falcon::VMachine vm;
   vm.call(0,0);
   vm.pushData(Falcon::Item());  // create an item -- local 0
   vm.pushCode( program );
   vm.run();

   vm.regA().toString( res );
   res.c_ize();
   std::cout << "Top: " << (char*)res.getRawStorage() << std::endl;

   return 0;
}
