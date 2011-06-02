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
#include <falcon/exprcompare.h>
#include <falcon/statement.h>

#include <falcon/synfunc.h>
#include <falcon/application.h>

class LoopApp: public Falcon::Application
{

public:
void go()
{
   Falcon::SynFunc fmain( "__main__" );
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


   Falcon::SynTree* program = &fmain.syntree();
   (*program)
      .append( new Falcon::StmtAutoexpr(new Falcon::ExprAssign( count->makeExpression(), new Falcon::ExprValue(0) ) ) )
      .append( new Falcon::StmtWhile(
                     new Falcon::ExprLT( count->makeExpression(), new Falcon::ExprValue(50000000) ),
                     assign ) );


   std::cout << program->describe().c_ize() << std::endl;

   // And now, run the code.
   Falcon::VMachine vm;
   vm.call(&fmain,0);
   vm.currentContext()->pushData(Falcon::Item());  // create an item -- local 0
   vm.run();

   std::cout << "Top: " << vm.regA().describe().c_ize() << std::endl;
}

};

// This is just a test.
int main( int argc, char* argv[] )
{
   std::cout << "Hello world" << std::endl;

   LoopApp loop;
   loop.go();
   return 0;
}
