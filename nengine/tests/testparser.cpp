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
#include <falcon/statement.h>
#include <falcon/stmtrule.h>
#include <falcon/rulesyntree.h>
#include <falcon/synfunc.h>
#include <falcon/extfunc.h>

#include <falcon/stdstreams.h>
#include <falcon/textwriter.h>



#include <falcon/trace.h>
#include <falcon/application.h>

#include <falcon/sourceparser.h>
#include <falcon/textreader.h>

#include "falcon/sourcelexer.h"

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

            vm->textOut()->write( *temp.asString() );
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



class ParserApp: public Falcon::Application
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
   // And now, run the code.
   Stream* my_stdin = new StdInStream(false);

   // the main
   SynTree MySynTree;
   TextReader* input = new TextReader(my_stdin);
   SourceParser parser(&MySynTree);

   SourceLexer* lexer = new SourceLexer("stdin", &parser, input);

   parser.pushLexer(lexer);
   ;

   if( parser.parse() )
   {
      std::cout << "Parsed code: "<< std::endl << MySynTree.describe().c_ize() << std::endl;
   }
   else
   {
      // errors:
      class MyEE: public Parsing::Parser::errorEnumerator {
      public:
         virtual bool operator()( const Parsing::Parser::ErrorDef& ed, bool blast )
         {
            std::cout << ed.nLine << ":" << ed.nChar << " - " << ed.nCode << " " << ed.sExtra.c_ize() << std::endl;
            return true;
         }
      } ee;

      std::cout << "ERRORS:"<< std::endl;
      parser.enumerateErrors( ee );
   }
}

};

// This is just a test.
int main( int argc, char* argv[] )
{
   std::cout << "Parser test!" << std::endl;

   TRACE_ON();

   ParserApp app;
   app.guardAndGo();

   return 0;
}

