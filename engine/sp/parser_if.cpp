/*
   FALCON - The Falcon Programming Language.
   FILE: parser_if.cpp

   Parser for Falcon source files -- if statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_if.cpp"

#include <falcon/setup.h>
#include <falcon/trace.h>

#include <falcon/error.h>
#include <falcon/statement.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_if.h>

#include <falcon/psteps/stmtif.h>
#include <falcon/psteps/exprvalue.h>

#include <falcon/synclasses_id.h>

namespace Falcon {

using namespace Parsing;

bool errhand_if(const NonTerminal&, Parser& p, int)
{
   TokenInstance* ti = p.getNextToken();
   TokenInstance* ti2 = p.getLastToken();

   if( p.lastErrorLine() < ti->line() )
   {
      // generate another error only if we didn't notice it already.
      p.addError( e_syn_if, p.currentSource(), ti->line(), ti2->chr() );
   }

   if( ! p.interactive() )
   {
      // put in a fake if statement (else, subsequent else/elif/end would also cause errors).
      SynTree* stIf = new SynTree;
      stIf->selector(new ExprValue(1));
      StmtIf* fakeIf = new StmtIf( stIf, ti->line(), ti->chr() );
      ParserContext* st = static_cast<ParserContext*>(p.context());
      st->openBlock( fakeIf, stIf );
   }
   else
   {
      // on interactive parsers, let the whole instruction to be destroyed.
      MESSAGE2( "errhand_if -- Ignoring IF in interactive mode." );
   }
   p.clearFrames();
   return true;
}


void apply_if_short( const NonTerminal&, Parser& p )
{
   // << (r_if_short << "if_short" << apply_if_short << T_if << Expr << T_Colon  )
   TokenInstance* tif = p.getNextToken();

   // don't open the if context if  we have an error in interactive mode.
   if( !p.interactive() || p.lastErrorLine() < tif->line() )
   {
      TokenInstance* texpr = p.getNextToken();      
      Expression* expr = static_cast<Expression*>(texpr->detachValue());
      ParserContext* ctx = static_cast<ParserContext*>(p.context());

      SynTree* ifTrue = new SynTree;
      ctx->accessSymbols(expr);
      ifTrue->selector(expr);
      StmtIf* stmt_if = new StmtIf(ifTrue, tif->line(), tif->chr());
      ctx->openBlock( stmt_if, ifTrue, true );
   }
   else
   {
      MESSAGE2( "apply_if_short -- Ignoring IF in interactive mode." );
   }
   
   // clear the stack
   p.simplify(3);
}


void apply_if( const NonTerminal&, Parser& p )
{
   // << (r_if << "if" << apply_if << T_if << Expr << T_EOL )
   TokenInstance* tif = p.getNextToken();

   // don't open the if context if  we have an error in interactive mode.
   if( !p.interactive() || p.lastErrorLine() < tif->line() )
   {
      ParserContext* ctx = static_cast<ParserContext*>(p.context());
      TokenInstance* texpr = p.getNextToken();
      p.getNextToken();

      Expression* expr = static_cast<Expression*>(texpr->detachValue());
      ParserContext* st = static_cast<ParserContext*>(p.context());

      SynTree* ifTrue = new SynTree;
      ctx->accessSymbols(expr);
      ifTrue->selector( expr );
      StmtIf* stmt_if = new StmtIf(ifTrue, tif->line(), tif->chr());
      st->openBlock( stmt_if, ifTrue );
   }
   else
   {
      MESSAGE2( "apply_if -- Ignoring IF in interactive mode." );
   }

   // clear the stack
   p.simplify(3);
}


void apply_elif( const NonTerminal&, Parser& p )
{
   // << (r_elif << "elif" << apply_elif << T_elif << Expr << T_EOL )
   TokenInstance* tif = p.getNextToken();

   // don't open the if context if  we have an error in interactive mode.
   if( !p.interactive() || p.lastErrorLine() < tif->line() )
   {
      TokenInstance* texpr = p.getNextToken();
      p.getNextToken();

      Expression* expr = static_cast<Expression*>(texpr->detachValue());
      ParserContext* st = static_cast<ParserContext*>(p.context());

      TreeStep* current = st->currentStmt();
      if( current == 0 || current->handler()->userFlags() != FALCON_SYNCLASS_ID_ELSEHOST )
      {
         p.addError( e_syn_elif, p.currentSource(), tif->line(), tif->chr() );
         delete expr;
      }
      else
      {
         StmtIf* stmt_if = static_cast<StmtIf*>(current);
         // can we really change branch now?
         SynTree* ifElse = st->changeBranch();
         if ( ifElse != 0 )
         {
            ParserContext* ctx = static_cast<ParserContext*>(p.context());
            ctx->accessSymbols(expr);
            ifElse->selector(expr);
            ifElse->decl( tif->line(), tif->chr() );
            stmt_if->addElif( ifElse );
         }
      }
   }
   else
   {
      MESSAGE2( "apply_elif -- Ignoring IF in interactive mode." );
   }
   
   // clear the stack
   p.simplify(3);
}


void apply_else( const NonTerminal&, Parser& p )
{
   // << (r_else << "else" << apply_else << T_else << T_EOL )
   TokenInstance* telse = p.getNextToken();

   // don't open the if context if  we have an error in interactive mode.
   if( !p.interactive() || p.lastErrorLine() < telse->line() )
   {
      p.getNextToken();

      ParserContext* st = static_cast<ParserContext*>(p.context());

      TreeStep* current = st->currentStmt();
      if( current == 0 || current->handler()->userFlags() != FALCON_SYNCLASS_ID_ELSEHOST )
      {
         p.addError( e_syn_else, p.currentSource(), telse->line(), telse->chr() );
      }
      else
      {
         StmtIf* stmt_if = static_cast<StmtIf*>(current);

         // can we really change branch?
         SynTree* ifElse = st->changeBranch();
         if( ifElse != 0 ) {
            ifElse->decl( telse->line(), telse->chr() );
            ifElse->selector(0); // just to be sure
            stmt_if->addElif(ifElse);
         }
      }
   }
   else
   {
      MESSAGE2( "apply_else -- Ignoring ELSE in interactive mode." );
   }

   // clear the stack
   p.simplify(2);
}

}

/* end of parser_if.cpp */
