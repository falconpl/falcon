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

#include <falcon/expression.h>
#include <falcon/statement.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/sp/parser_if.h>

namespace Falcon {

using namespace Parsing;

void apply_if_short( const Rule&, Parser& p )
{
   // << (r_if_short << "if_short" << apply_if_short << T_if << Expr << T_Colon << S_Autoexpr << T_EOL )

   TokenInstance* tif = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();
   p.getNextToken();
   TokenInstance* tstatement = p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   Expression* sa = static_cast<Expression*>(tstatement->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   SynTree* ifTrue = new SynTree;
   ifTrue->append( new StmtAutoexpr(sa) );

   StmtIf* stmt_if = new StmtIf(expr, ifTrue, 0, tif->line(), tif->chr());
   st->addStatement( stmt_if );

   // clear the stack
   p.simplify(5);
}


void apply_if( const Rule&, Parser& p )
{
   // << (r_if << "if" << apply_if << T_if << Expr << T_EOL )
   TokenInstance* tif = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();
   p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   SynTree* ifTrue = new SynTree;
   StmtIf* stmt_if = new StmtIf(expr, ifTrue, 0, tif->line(), tif->chr());
   st->openBlock( stmt_if, ifTrue );

   // clear the stack
   p.simplify(3);
}


void apply_elif( const Rule&, Parser& p )
{
   // << (r_elif << "elif" << apply_elif << T_elif << Expr << T_EOL )
   TokenInstance* tif = p.getNextToken();
   TokenInstance* texpr = p.getNextToken();
   p.getNextToken();

   Expression* expr = static_cast<Expression*>(texpr->detachValue());
   ParserContext* st = static_cast<ParserContext*>(p.context());

   Statement* current = st->currentStmt();
   if( current == 0 || current->type() != Statement::if_t )
   {
      p.addError( e_syn_elif, p.currentSource(), tif->line(), tif->chr() );
      delete expr;
   }
   else
   {
      StmtIf* stmt_if = static_cast<StmtIf*>(current);
      // can we really change branch now?
      SynTree* ifElse = st->changeBranch();
      if ( ifElse != 0 ) {
         stmt_if->addElif( expr, ifElse, tif->line(), tif->chr() );
      }
   }

   // clear the stack
   p.simplify(3);
}


void apply_else( const Rule&, Parser& p )
{
   // << (r_else << "else" << apply_else << T_else << T_EOL )
   TokenInstance* telse = p.getNextToken();
   p.getNextToken();

   ParserContext* st = static_cast<ParserContext*>(p.context());

   Statement* current = st->currentStmt();
   if( current == 0 || current->type() != Statement::if_t )
   {
      p.addError( e_syn_else, p.currentSource(), telse->line(), telse->chr() );
   }
   else
   {
      StmtIf* stmt_if = static_cast<StmtIf*>(current);

      // can we really change branch?
      SynTree* ifElse = st->changeBranch();
      if( ifElse != 0 ) {
         stmt_if->setElse( ifElse );
      }
   }

   // clear the stack
   p.simplify(2);
}

}

/* end of parser_if.cpp */
