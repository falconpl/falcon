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
#include <falcon/synfunc.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/sp/parser_class.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

static void make_class( Parser& p, int tCount, TokenInstance* tname, TokenInstance* tParams )
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   // Are we already in a function?
   if( ctx->currentFunc() != 0 || ctx->currentClass() != 0 || ctx->currentStmt() != 0 )
   {
      p.addError( e_toplevel_class,  p.currentSource(), tname->line(), tname->chr() );
      p.simplify( tCount  );
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
      p.simplify( tCount );
      return;
   }

   // Ok, we took the symbol.
   FalconClass* cls = new FalconClass( *tname->asString() );

   // Some parameters to take care of?
   if( tParams != 0 )
   {
      NameList* list = static_cast<NameList*>( tParams->asData() );
      cls->setInit( new SynFunc( "init" ));

      for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
      {
         cls->init()->symbols().addLocal( *it );
      }
   }

   // remove this stuff from the stack
   p.simplify( tCount );

   ctx->openClass(cls, false, symclass);
   p.pushState( "ClassBody" );
}


void apply_class( const Rule&, Parser& p )
{
   // << T_class << T_Name << T_EOL
   p.getNextToken(); // T_class
   TokenInstance* tname=p.getNextToken();

   make_class(p, 3, tname, 0 );
}




void apply_class_p( const Rule&, Parser& p )
{
   // << T_class << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL
   p.getNextToken(); // T_class
   TokenInstance* tname = p.getNextToken(); // T_Name
   p.getNextToken(); // T_Openpar
   TokenInstance* tparams = p.getNextToken(); // ListSymbol

   make_class( p, 6, tname, tparams );
}


void apply_pdecl_expr( const Rule&, Parser& p )
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
      cls->addProperty( *tname->asString(), expr );
   }

   // time to add the things we have parsed.
   ctx->checkSymbols();

   // remove this stuff from the stack
   p.simplify( 4 );
}

void apply_init_expr( const Rule&, Parser& p )
{
   // T_init << EOL;
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   // we should be in class state.
   FalconClass* cls = (FalconClass*) ctx->currentClass();
   fassert( cls != 0 );

   if( cls->hasInit() )
   {
      TokenInstance* ti = p.getNextToken();
      p.addError( e_init_given, p.currentSource(), ti->line(), ti->chr() );
      p.simplify( 2 );
   }
   else
   {
      cls->hasInit(true);
      // but.. we don't really know if the init funciton has been created
      if( cls->init() == 0 )
      {
         cls->setInit( new SynFunc("init") );
      }

      // the user can add a non-syntree function as init,
      // ... but we won't be here compiling it.
      ctx->openFunc( static_cast<SynFunc*>(cls->init()) );
      p.simplify( 2 );
      p.pushState("Main");
   }

}

}

/* end of parser_class.cpp */
