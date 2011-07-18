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

#include <falcon/expression.h>
#include <falcon/exprarray.h>
#include <falcon/exprdict.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>

#include "private_types.h"
#include <falcon/sp/parser_arraydecl.h>

namespace Falcon {

using namespace Parsing;

void apply_expr_array_decl( const Rule&, Parser& p )
{
   // << (r_Expr_index << "Expr_array_decl" << apply_expr_array_decl << T_OpenSquare << ListExprOrPairs << T_CloseSquare )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tSquare = p.getNextToken();
   TokenInstance* v1 = p.getNextToken();
   p.getNextToken();

   TokenInstance* ti = new TokenInstance(tSquare->line(), tSquare->chr(), sp.Expr);

   PairList* list = static_cast<PairList*>(v1->detachValue());
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
      ti->setValue( dict, expr_deletor );
   }
   else
   {
      // it's an array declaration
      ExprArray* array = new ExprArray;
      PairList::iterator iter = list->begin();
      while( iter != list->end() )
      {
         // Todo -- make the array expression
         array->add(iter->first);
         ++iter;
      }
      ti->setValue( array, expr_deletor );
   }

   // free the expressions in the list
   list->clear();

   p.simplify(3,ti);
}

void apply_expr_empty_dict( const Rule&, Parser& p )
{
   // << T_OpenSquare << T_Arrow << T_CloseSquare )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tSquare = p.getNextToken();
   TokenInstance* ti = new TokenInstance(tSquare->line(), tSquare->chr(), sp.Expr);
   ExprDict* dict = new ExprDict;
   ti->setValue( dict, expr_deletor );
   p.simplify(3,ti);
}

}

/* end of parser_arraydecl.cpp */
