/*
   FALCON - The Falcon Programming Language.
   FILE: parser_arraydecl.cpp

   Parser for Falcon source files -- handler for array and dict decl
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_arraydecl.cpp"

#include <falcon/setup.h>
#include <falcon/error.h>

#include <falcon/psteps/exprarray.h>
#include <falcon/psteps/exprdict.h>
#include <falcon/psteps/exprrange.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>

#include "private_types.h"
#include <falcon/sp/parser_arraydecl.h>

namespace Falcon {

using namespace Parsing;


bool ArrayEntry_errHand( const NonTerminal&, Parser&p, int )
{
   p.setErrorMode(&p.T_EOL);
   return false;
}

static Expression* make_array_expr( PairList* list )
{
   Expression* retval;

   if( list->m_bHasPairs )
   {
      // It's a dictionary declaration.
      ExprDict* dict = new ExprDict;
      PairList::iterator iter = list->begin();
      while( iter != list->end() )
      {
         // Todo -- make the array expression
         dict->add(iter->first, iter->second);
         ++iter;
      }

      retval = dict;
   }
   else
   {
      // it's an array declaration
      ExprArray* array = new ExprArray;

      PairList::iterator iter = list->begin();
      while( iter != list->end() )
      {
         // Todo -- make the array expression
         array->append(iter->first);
         ++iter;
      }
      retval = array;
   }

   // free the expressions in the list
   list->clear();
   delete list;
   return retval;
}


// temporary statement used to keep track of the forming array expression
class StmtTempArrayDecl: public Statement
{
public:
   PairList* m_forming;
   Expression* m_FirstExpr;
   bool bFirstGiven;
   bool bHasPair;
   bool bHasSep;
   int definedAt;

   typedef enum
   {
      first_expr,
      separator,
      arrow,
      second_expr,
      closing
   }
   t_state;

   t_state state;

   StmtTempArrayDecl():
      Statement( 0,0 ),
      m_forming( new PairList ),
      m_FirstExpr(0),
      bFirstGiven( false ),
      bHasPair( false ),
      bHasSep( false ),
      definedAt(0),
      state( first_expr )
   {
      // don't record us, we're temp.
      m_discardable = true;
   }

   ~StmtTempArrayDecl()
   {
      if( m_forming != 0)
      {
         pair_list_deletor( m_forming );
      }
      delete m_FirstExpr;
   }

   void advance_state()
   {
      switch( this->state )
      {
         case first_expr:
            // -- need a comma?
            if( this->bHasPair )
            {
               // no, we always need an arrow
               this->state = arrow;
            }
            else if( this->bHasSep )
            {
               this->state = separator;
            }
            // else stay in first expr.
            break;

         // after a comma we always need a first expr
         case separator:
            this->state = first_expr;
            break;

         // after an arrow, need a second expr
         case arrow:
            this->state = second_expr;
            break;

         case second_expr:
            // -- need a comma?
            if( this->bHasSep )
            {
               // no, we always need an arrow
               this->state = separator;
            }
            else
            {
               this->state = first_expr;
            }
            break;

         case closing:
            // stay closing
            break;
      }
   }

   void render( TextWriter*, int ) const { }

   virtual StmtTempArrayDecl* clone() const { return 0; }
};


static StmtTempArrayDecl* internal_apply_expr_array_decl( const NonTerminal&, Parser& p )
{
   // << T_OpenSquare
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());

   // our return will be an expression
   TokenInstance* openpar = p.getNextToken();

   StmtTempArrayDecl* decl = new StmtTempArrayDecl;
   decl->definedAt = openpar->line();
   ctx->openBlock( decl, 0 );
   p.pushState( "ArrayDecl" );

   return decl;
}


void apply_expr_array_decl( const NonTerminal& r, Parser& p )
{
   internal_apply_expr_array_decl( r, p )->bHasSep = true;
}


void apply_expr_array_decl2( const NonTerminal& r, Parser& p )
{
   internal_apply_expr_array_decl( r, p )->bHasSep = false;
}


void apply_array_entry_expr( const NonTerminal&, Parser&p )
{
   // Expr
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   StmtTempArrayDecl* decl = static_cast<StmtTempArrayDecl*>(ctx->currentStmt());
   fassert( decl != 0 );

   TokenInstance* texpr = sp.getNextToken();
   Expression* expr = static_cast<Expression*>(texpr->detachValue());

   // are we waiting for an expression?
   if( decl->state == StmtTempArrayDecl::first_expr )
   {
      if( decl->m_FirstExpr != 0 )
      {
         Expression* second = 0;
         decl->m_forming->push_back( std::make_pair<Expression*, Expression*>(decl->m_FirstExpr, second ) );
         decl->bFirstGiven = true;
      }

      decl->m_FirstExpr = expr;
      decl->advance_state();
   }
   else if ( decl->state == StmtTempArrayDecl::second_expr )
   {
      // we wouldn't be in this state if not authorized.
      decl->m_forming->push_back( std::make_pair<Expression*, Expression*>(decl->m_FirstExpr, expr) );
      decl->m_FirstExpr = 0;
      decl->bFirstGiven = true;
      decl->advance_state();
   }
   else
   {
      p.addError( decl->bHasPair ? e_syn_dictdecl : e_syn_arraydecl,
         p.currentSource(), texpr->line(), texpr->chr(), decl->definedAt );
      delete expr;
   }

    p.simplify(1);
}


void apply_array_entry_comma( const NonTerminal&, Parser& p )
{
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   StmtTempArrayDecl* decl = static_cast<StmtTempArrayDecl*>(ctx->currentStmt());
   fassert( decl != 0 );

   // When we expect a comma, we can only have a comma.
   if( decl->state != StmtTempArrayDecl::separator )
   {
      // but, we can accept a comma if we're waiting for a first in a .[ list
      if( ! (decl->state == StmtTempArrayDecl::first_expr && ! decl->bHasSep ) )
      {
         TokenInstance* ti = sp.getNextToken();
         p.addError( decl->bHasPair ? e_syn_dictdecl : e_syn_arraydecl,
            p.currentSource(), ti->line(), ti->chr(), decl->definedAt, "expecting expression" );
      }
   }
   else
   {
      decl->advance_state();
   }

   p.simplify(1);
}


void apply_array_entry_eol( const NonTerminal&, Parser& p )
{
   // ignored.
   p.simplify(1);
}


void apply_array_entry_arrow( const NonTerminal&, Parser& p )
{
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   StmtTempArrayDecl* decl = static_cast<StmtTempArrayDecl*>(ctx->currentStmt());
   fassert( decl != 0 );

   // waiting for an arrow?
   if( decl->state == StmtTempArrayDecl::arrow )
   {
      decl->advance_state();
   }
   else
   {
      // no? -- either we're at the very beginning...
      if( decl->m_FirstExpr == 0 )
      {
         // then we have a [=> and must wait for ]
         decl->state = StmtTempArrayDecl::closing;
         decl->m_forming->m_bHasPairs = true;
      }
      else
      {
         //... or we have the first, but we have not committed it
         if( ! decl->bFirstGiven )
         {
            // so we have [ first =>  and need to wait for the second.
            decl->bHasPair = true;
            decl->m_forming->m_bHasPairs = true;
            decl->state = StmtTempArrayDecl::second_expr;
         }
         else
         {
            // we definitely have an error.
            TokenInstance* ti = sp.getNextToken();
            p.addError( decl->bHasPair ? e_syn_dictdecl : e_syn_arraydecl,
               p.currentSource(), ti->line(), ti->chr(), decl->definedAt, "misplaced =>" );
         }
      }
   }

   p.simplify(1);
}


void apply_array_entry_close( const NonTerminal&, Parser& p )
{
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   StmtTempArrayDecl* decl = static_cast<StmtTempArrayDecl*>(ctx->currentStmt());
   fassert( decl != 0 );

   // if waiting for an arrow or a second, we have an error.
   if( decl->state == StmtTempArrayDecl::second_expr
      || decl->state == StmtTempArrayDecl::arrow
      || ( decl->state == StmtTempArrayDecl::first_expr && decl->bHasSep && decl->arity() != 0 )
      )
   {
      TokenInstance* ti = p.getNextToken();
      p.addError( decl->bHasPair ? e_syn_dictdecl : e_syn_arraydecl,
         p.currentSource(), ti->line(), ti->chr(), decl->definedAt, "closing too early" );
   }
   else
   {
      // but if we wait for a comma, we can get the generated first
      if( decl->m_FirstExpr )
      {
         decl->m_forming->push_back( std::make_pair<Expression*, Expression*>(decl->m_FirstExpr,0) );
         decl->m_FirstExpr = 0;
      }
   }

   // anyhow, we're out of business.
   PairList* pl = decl->m_forming;
   decl->m_forming = 0;

   p.simplify(1);
   ctx->closeContext();

   // we still have the [ in the stack.
   TokenInstance* openPar = p.getLastToken();
   // close the list
   Expression* made = make_array_expr( pl );
   made->decl( openPar->line(), openPar->chr() );
   openPar->setValue( made, treestep_deletor );
   openPar->token( sp.Expr );
}

static void makeRange( Parser& p, int count, Expression* expr1, Expression* expr2, Expression* expr3 )
{
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   StmtTempArrayDecl* decl = static_cast<StmtTempArrayDecl*>(ctx->currentStmt());
   fassert( decl != 0 );

   // We can't close a range if we're not in first state.
   if( decl->m_forming->size() != 0 || decl->m_FirstExpr != 0 )
   {
      TokenInstance* ti = p.getNextToken();
      p.addError( e_syn_rangedecl,
         p.currentSource(), ti->line(), ti->chr(), decl->definedAt, "mixed range in array" );
   }

   // anyhow, we're out of business -- and we can discard the forming data.
   p.simplify(count);
   ctx->closeContext();
   Expression* made = new ExprRange( expr1, expr2, expr3 );

   // we still have the [ in the stack.
   TokenInstance* openPar = p.getLastToken();
   // close the list
   made->decl( openPar->line(), openPar->chr() );
   openPar->setValue( made, treestep_deletor );
   openPar->token( sp.Expr );
}


void apply_array_entry_range3( const NonTerminal&, Parser& p )
{
   // << Expr << T_Colon << Expr << T_Colon << Expr << T_CloseSquare

   TokenInstance* texpr1 = p.getNextToken();
   p.getNextToken(); //:
   TokenInstance* texpr2 = p.getNextToken();
   p.getNextToken(); //:
   TokenInstance* texpr3 = p.getNextToken();

   makeRange( p, 6,
         static_cast<Expression*>(texpr1->detachValue()),
         static_cast<Expression*>(texpr2->detachValue()),
         static_cast<Expression*>(texpr3->detachValue())
      );
}


void apply_array_entry_range3bis( const NonTerminal&, Parser& p )
{
   // << Expr << T_Colon << T_Colon << Expr << T_CloseSquare
   TokenInstance* texpr1 = p.getNextToken();
   p.getNextToken(); //:
   p.getNextToken(); //:
   TokenInstance* texpr3 = p.getNextToken();

   makeRange( p, 5,
         static_cast<Expression*>(texpr1->detachValue()),
         0,
         static_cast<Expression*>(texpr3->detachValue())
      );
}


void apply_array_entry_range2( const NonTerminal&, Parser& p )
{
   // << Expr << T_Colon << Expr << T_CloseSquare
   TokenInstance* texpr1 = p.getNextToken();
   p.getNextToken(); //:
   TokenInstance* texpr2 = p.getNextToken();
   makeRange( p, 4,
         static_cast<Expression*>(texpr1->detachValue()),
         static_cast<Expression*>(texpr2->detachValue()),
         0
      );

}


void apply_array_entry_range1( const NonTerminal&, Parser& p )
{
   // << Expr << T_Colon << T_CloseSquare
   TokenInstance* texpr1 = p.getNextToken();
   makeRange( p, 3,
         static_cast<Expression*>(texpr1->detachValue()),
         0,
         0
      );
}


void apply_array_entry_runaway( const NonTerminal&, Parser& )
{

}

}

/* end of parser_arraydecl.cpp */
