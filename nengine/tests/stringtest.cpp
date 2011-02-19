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
#include <falcon/synfunc.h>
#include <falcon/extfunc.h>

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
   // str = string_add( "A string" )
   // printl( str )

   SynFunc string_add( "string_add" );
   Symbol* param = string_add.addVariable("param");
   string_add.paramCount(1);

   string_add.syntree().append(
         new StmtReturn( new ExprPlus( param->makeExpression(), new ExprValue( " -- Hello world" ) ) )
   );

   // and now the main function
   ExprCall* call_func = new ExprCall( new ExprValue(&string_add) );
   call_func->addParameter( new ExprValue("A string") );


   // and the main
   SynFunc fmain( "__main__" );
   Falcon::Symbol* strsym = new LocalSymbol("str",0);
   fmain.syntree()
      .append( new StmtAutoexpr(
               new ExprAssign( strsym->makeExpression(), call_func ) ))
      .append( new StmtAutoexpr(&(new ExprCall( new ExprValue(&printl) ))
            ->addParameter(strsym->makeExpression()).addParameter(new ExprValue(1))) )
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
int main( int argc, char* argv[] )
{
   std::cout << "string test!" << std::endl;

   TRACE_ON();
   
   StringApp app;
   app.guardAndGo();

   return 0;
}
