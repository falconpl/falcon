/*
   FALCON - The Falcon Programming Language.
   FILE: parser_class.cpp

   Parser for Falcon source files -- class statement handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 11 Apr 2011 00:04:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_class.cpp"

#include <falcon/setup.h>

#include <falcon/expression.h>
#include <falcon/exprvalue.h>
#include <falcon/globalsymbol.h>
#include <falcon/falconclass.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/sp/parser_class.h>

namespace Falcon {

using namespace Parsing;

void apply_class( const Rule&, Parser& p )
{
   // << T_class << T_Name << T_EOL
   //static Class* cc = Engine::instance()->classClass();

   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   sp.getNextToken(); // T_class
   TokenInstance* tname=sp.getNextToken();
   sp.getNextToken();// 'EOL'

   // Are we already in a function?
   if( ctx->currentFunc() != 0 || ctx->currentClass() != 0 || ctx->currentStmt() != 0 )
   {
      p.addError( e_toplevel_class,  p.currentSource(), tname->line(), tname->chr() );
      p.simplify(3);
      return;
   }

   // check if the symbol is free -- defining an unique symbol
   bool alreadyDef;
   GlobalSymbol* symclass = ctx->onGlobalDefined( *tname->asString(), alreadyDef );
   if( alreadyDef )
   {
      // not free!
      p.addError( e_already_def,  p.currentSource(), tname->line(), tname->chr(), 0,
         String("at line ").N(symclass->declaredAt()) );
      p.simplify(3);
      return;
   }

   // Ok, we took the symbol.
   Class* cls = new FalconClass( *tname->asString() );

   // remove this stuff from the stack
   p.simplify( 3 );

   ctx->openClass(cls, false, symclass);
   p.pushState( "ClassBody", 0 , &p );
}


void apply_class_p( const Rule&, Parser&  )
{
 // << T_class << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL

   //p.simplify( 6 );
}


void apply_pdecl_expr( const Rule&, Parser& p  )
{
   // << T_Name << T_EqSign << Expr << T_EOL;
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   // we should be in class state.
   FalconClass* cls = (FalconClass*) ctx->currentClass();
   fassert( cls != 0 );

   TokenInstance* tname = sp.getNextToken(); // T_Name
   sp.getNextToken(); // =
   TokenInstance* texpr = sp.getNextToken();
   sp.getNextToken(); // 'EOL'

   Expression* expr = (Expression*) texpr->detachValue();
   if( expr->type() == Expression::t_value )
   {
      cls->addProperty( *tname->asString(), static_cast<ExprValue*>(expr)->item() );
      // we don't need the expression anymore
      delete expr;
   }
   else
   {
      cls->addProperty( *tname->asString(), Item() );
      // todo -- add the expression to init
   }
   // remove this stuff from the stack
   p.simplify( 4 );
}

}

/* end of parser_class.cpp */
