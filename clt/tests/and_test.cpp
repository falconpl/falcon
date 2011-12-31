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
#include <falcon/symbol.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <falcon/statement.h>
#include <falcon/synfunc.h>
#include <falcon/extfunc.h>

#include <falcon/synfunc.h>
#include <falcon/module.h>
#include <falcon/application.h>

#include <falcon/psteps/stmtif.h>
#include <falcon/psteps/stmtautoexpr.h>
#include <falcon/psteps/exprcall.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/psteps/exprmath.h>
#include <falcon/psteps/exprsym.h>
#include <falcon/psteps/exprassign.h>
#include <falcon/psteps/exprlogic.h>

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

      static void apply_( const PStep* ps, VMContext* ctx )
      {
         const NextStep* nstep = static_cast<const NextStep*>(ps);
         std::cout << *ctx->regA().asString()->c_ize();
         nstep->printNext( ctx, ctx->currentCode().m_seqId );
      }

      void printNext( VMContext* ctx, int count ) const
      {
         int nParams = ctx->currentFrame().m_paramCount;

         ctx->condPushCode( this );
         while( count < nParams )
         {
            Class* cls;
            void* data;

            ctx->param(count)->forceClassInst( cls, data );
            ++count;
            ctx->currentCode().m_seqId = count;

            ctx->pushData(*ctx->param(count));
            cls->op_toString( ctx, data );
            if( ctx->wentDeep(this) )
            {               
               return;
            }
            std::cout << ctx->topData().asString()->c_ize();
            ctx->popData();
         }
         ctx->popCode();
         
         std::cout << std::endl;
         // we're out of the function.
         ctx->returnFrame();
      }
   } m_nextStep;

   FuncPrintl():
      Function("printl")
   {}

   virtual ~FuncPrintl() {}

   virtual void invoke( VMContext* ctx, int32)
   {
      m_nextStep.printNext( ctx, 0 );
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

   
   Symbol* count = new Symbol( "count", Symbol::e_st_local, 0);
   Expression* assign = new ExprAssign( new ExprSymbol( count ),
                     new ExprPlus( new ExprValue(2), new ExprValue(1) ));

   SynTree* iftrue = new SynTree;
      iftrue->append( new StmtAutoexpr(
            &(*(new ExprCall( new ExprValue(&printl) ))).add(new ExprValue("TRUE:")).add(new ExprSymbol(count)))
             );

   SynTree* iffalse = new SynTree;
      iffalse->append( new StmtAutoexpr(
            &(*(new ExprCall( new ExprValue(&printl) ))).add(new ExprValue("FALSE:")).add(new ExprSymbol(count)))
             );

   Expression* check = bUseOr ?
         static_cast<Expression*>(new ExprOr( new ExprValue(arg), assign )):
         static_cast<Expression*>(new ExprAnd( new ExprValue(arg), assign ));

   SynTree* program = &fmain.syntree();
   iftrue->selector(check);
   (*program)
      .append( new StmtIf( iftrue, iffalse ) );
   
   std::cout << program->describe().c_ize() << std::endl;

   // And now, run the code.
   Falcon::VMachine vm;
   vm.currentContext()->call(&fmain,0);
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
