/*
 * main.cpp
 *
 *  Created on: 12/gen/2011
 *      Author: gian
 */

#include <cstdio>
#include <iostream>
#include <string>

#include <falcon/vm.h>
#include <falcon/syntree.h>
#include <falcon/localsymbol.h>
#include <falcon/error.h>
#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/exprsym.h>
#include <falcon/statement.h>
#include <falcon/synfunc.h>
#include <falcon/extfunc.h>


#include <falcon/synfunc.h>
#include <falcon/module.h>
#include <falcon/application.h>

using namespace Falcon;


class FuncPrintl: public Function
{
public:
   class NextStep: public PStep
   {
   public:
      NextStep()
      {
         apply = apply_;
      }

      static void apply_( const PStep* ps, VMachine* vm )
      {
         const NextStep* nstep = static_cast<const NextStep*>(ps);
         std::cout << *vm->regA().asString()->c_ize();
         VMContext* ctx = vm->currentContext();
         nstep->printNext( vm, ctx->currentCode().m_seqId );
      }

      void printNext( VMachine* vm, int count ) const
      {
         VMContext* ctx = vm->currentContext();
         int nParams = ctx->currentFrame().m_paramCount;

         while( count < nParams )
         {
            Item temp;
            Class* cls;
            void* data;

            ctx->param(count)->forceClassInst( cls, data );
            ++count;

            vm->ifDeep(this);
            cls->op_toString( vm, data, temp );
            if( vm->wentDeep() )
            {
               ctx->currentCode().m_seqId = count;
               return;
            }
            std::cout << temp.asString()->c_ize();
         }

         std::cout << std::endl;
         // we're out of the function.
         vm->returnFrame();
      }
   } m_nextStep;

   FuncPrintl():
      Function("printl")
   {}

   virtual ~FuncPrintl() {}

   virtual void apply( VMachine* vm, int32 nParams )
   {
      m_nextStep.printNext( vm, 0 );
   }
} printl;



class DebugTestApp: public Falcon::Application
{

public:
void go( int arg, bool bUseOr )
{
   Falcon::Module module("andtest", "./andtest.cpp");
   Falcon::SynFunc fmain( "__main__" );
   fmain.module(&module);

   
   Symbol* count = new LocalSymbol("count",0);
   Expression* assign = new ExprAssign( count->makeExpression(),
                     new ExprPlus( new ExprValue(2), new ExprValue(1) ));

   SynTree* iftrue = new SynTree;
      iftrue->append( new StmtAutoexpr(
            &(*(new ExprCall( new ExprValue(&printl) ))).addParameter(new ExprValue("TRUE:")).addParameter(count->makeExpression()))
             );

   SynTree* iffalse = new SynTree;
      iffalse->append( new StmtAutoexpr(
            &(*(new ExprCall( new ExprValue(&printl) ))).addParameter(new ExprValue("FALSE:")).addParameter(count->makeExpression()))
             );

   Expression* check = bUseOr ?
         static_cast<Expression*>(new ExprOr( new ExprValue(arg), assign )):
         static_cast<Expression*>(new ExprAnd( new ExprValue(arg), assign ));

   SynTree* program = &fmain.syntree();
   (*program)
      .append( new StmtIf( check, iftrue, iffalse ) );
   
   std::cout << program->describe().c_ize() << std::endl;

   // And now, run the code.
   Falcon::VMachine vm;
   vm.call(&fmain,0);
   vm.currentContext()->pushData(Falcon::Item());  // create an item -- local 0

   while( vm.nextStep() != 0 )
   {
      std::string choice;

      Falcon::String status = vm.location();
      std::cout << status.c_ize() << " - " << vm.nextStep()->oneLiner().c_ize()
            << "..."<< std::endl;
      std::cout << vm.report().c_ize() << std::endl;

      std::cin >> choice;

      if ( choice == "quit" )
      {
         break;
      }
      else if( choice == "step" || choice == "s" )
      {
         vm.step();
      }
      else if( choice == "run" || choice == "r" )
      {
         vm.run();
      }
   }
   
   std::cout << "Top: " << vm.regA().describe().c_ize() << std::endl;
   }
};

// This is just a test.
int main( int argc, char* argv[] )
{
   std::cout << "Debug test" << std::endl;

   DebugTestApp gotest;
   gotest.go( argc < 2 ? 0 : atoi(argv[1]), argc > 2 );
   return 0;
}
