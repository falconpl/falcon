/*
 * ruletest.cpp
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
#include <falcon/exprcompare.h>
#include <falcon/exprmath.h>
#include <falcon/exprcall.h>
#include <falcon/statement.h>
#include <falcon/stmtrule.h>
#include <falcon/rulesyntree.h>
#include <falcon/synfunc.h>
#include <falcon/extfunc.h>

#include <falcon/stdstreams.h>
#include <falcon/textwriter.h>



#include <falcon/trace.h>
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
         vm->textOut()->write( *vm->regA().asString() );
         VMContext* ctx = vm->currentContext();
         nstep->printNext( vm, ctx->currentCode().m_seqId );
      }

      void printNext( VMachine* vm, int count ) const
      {
         VMContext* ctx = vm->currentContext();
         int nParams = ctx->currentFrame().m_paramCount;

         while( count < nParams )
         {
            Class* cls;
            void* data;

            ctx->param(count)->forceClassInst( cls, data );
            ++count;

            ctx->pushData(*ctx->param(count));
            vm->ifDeep(this);
            cls->op_toString( vm, data );
            if( vm->wentDeep() )
            {
               ctx->currentCode().m_seqId = count;
               return;
            }
            
            vm->textOut()->write( *ctx->topData().asString() );
            ctx->popData();
         }

         vm->textOut()->write( "\n" );
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



class RuleApp: public Falcon::Application
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
   // rule
   //   a = 0
   //   ? a = a + 1
   //   printl( a )
   //   a >= 10
   // end
   // printl( "A was ", a )
   //


   // the main
   SynFunc fmain( "__main__" );
   Falcon::Symbol* var_a = fmain.symbols().addLocal("a");

   StmtAutoexpr* assign_expr = new StmtAutoexpr(
               new ExprAssign( var_a->makeExpression(),
                  new ExprPlus( var_a->makeExpression(), new ExprValue(1) ) ) );
   assign_expr->nd(true);
   // and the rule
   StmtRule* rule = new StmtRule;
   (*rule)
         .addStatement( new StmtAutoexpr(
               new ExprAssign( var_a->makeExpression(), new ExprValue(0) ) ) )
         .addStatement( assign_expr )
         .addStatement( new StmtAutoexpr(&(new ExprCall( new ExprValue(&printl) ))
            ->addParam(new ExprValue("A: ")).addParam(var_a->makeExpression())) )
         .addStatement( new StmtAutoexpr(
               new ExprGE( var_a->makeExpression(), new ExprValue(10) ) ) );

   fmain.syntree()
      .append( rule )
      .append( new StmtAutoexpr(&(new ExprCall( new ExprValue(&printl) ))
            ->addParam(new ExprValue("A was ")).addParam(var_a->makeExpression())) );
      

   std::cout << "Will run: "<< std::endl << fmain.syntree().describe().c_ize() << std::endl;

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
   std::cout << "Rule test!" << std::endl;

   TRACE_ON();

   RuleApp app;
   app.guardAndGo();

   return 0;
}

