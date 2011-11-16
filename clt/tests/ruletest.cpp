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
#include <falcon/error.h>
#include <falcon/statement.h>
#include <falcon/rulesyntree.h>
#include <falcon/synfunc.h>
#include <falcon/extfunc.h>

#include <falcon/stdstreams.h>
#include <falcon/textwriter.h>

#include <falcon/trace.h>
#include <falcon/application.h>

#include <falcon/psteps/stmtautoexpr.h>
#include <falcon/psteps/stmtrule.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/psteps/exprsym.h>
#include <falcon/psteps/exprcompare.h>
#include <falcon/psteps/exprmath.h>
#include <falcon/psteps/exprcall.h>

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
         ctx->vm()->textOut()->write( *ctx->regA().asString() );
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
            if( ctx->wentDeep( this ) )
            {
               return;
            }
            
            ctx->vm()->textOut()->write( *ctx->topData().asString() );
            ctx->popData();
         }
         ctx->popCode();

         ctx->vm()->textOut()->write( "\n" );
         // we're out of the function.
         ctx->returnFrame();
      }
   } m_nextStep;

   FuncPrintl():
      Function("printl")
   {}

   virtual ~FuncPrintl() {}

   virtual void invoke( VMContext* ctx, int32 )
   {
      m_nextStep.printNext( ctx, 0 );
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
               new ExprAssign( new ExprSymbol(var_a),
                  new ExprPlus( new ExprSymbol(var_a), new ExprValue(1) ) ) );
   
   // and the rule
   StmtRule* rule = new StmtRule;
   (*rule)
         .addStatement( new StmtAutoexpr(
               new ExprAssign( new ExprSymbol(var_a), new ExprValue(0) ) ) )
         .addStatement( assign_expr )
         .addStatement( new StmtAutoexpr(&(new ExprCall( new ExprValue(&printl) ))
            ->addParam( new ExprValue("A: ")).addParam(new ExprSymbol(var_a))) )
         .addStatement( new StmtAutoexpr(
               new ExprGE( new ExprSymbol(var_a), new ExprValue(10) ) ) );

   fmain.syntree()
      .append( rule )
      .append( new StmtAutoexpr(&(new ExprCall( new ExprValue(&printl) ))
            ->addParam(new ExprValue("A was ")).addParam(new ExprSymbol(var_a))) );
      

   std::cout << "Will run: "<< std::endl << fmain.syntree().describe().c_ize() << std::endl;

   // And now, run the code.
   Falcon::VMachine vm;
   vm.currentContext()->call(&fmain,0);
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
   std::cout << "Rule test!" << std::endl;

   TRACE_ON();

   RuleApp app;
   app.guardAndGo();

   return 0;
}

