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
#include <falcon/trace.h>

#include <falcon/symbol.h>
#include <falcon/expression.h>
#include <falcon/error.h>
#include <falcon/falconclass.h>
#include <falcon/synfunc.h>

#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/sp/parser_class.h>
#include <falcon/parser/lexer.h>

#include <falcon/psteps/exprvalue.h>
#include <falcon/psteps/exprinherit.h>
#include <falcon/psteps/exprparentship.h>

#include "private_types.h"

#include <list>

namespace Falcon {


bool classdecl_errhand(const NonTerminal&, Parser& p, int )
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* ti = p.getNextToken();
   
   // already detected?
   if( p.lastErrorLine() != ti->line() )
   {
      p.addError( e_syn_class, p.currentSource(), ti->line(), ti->chr() );
   }
   
   // remove the whole line
   p.setErrorMode(&p.T_EOL);
   
   if( ! p.interactive() )
   {
      // put in a fake if statement (else, subsequent else/elif/end would also cause errors).
      FalconClass* cls = new FalconClass( "" );
      ctx->openClass( cls, false );
      p.pushState("ClassBody");
   }
   else
   {
      MESSAGE2( "classdecl_errhand -- Ignoring CLASS in interactive mode." );
      Class* cls = ctx->currentClass();
      if( cls != 0 ) {
         ctx->dropContext();
         p.popState();
         delete cls;
      }
   }
      
   // we need to create a discardable anonymous class if we're a module.   
   return true;
}

using namespace Parsing;

static void make_class( Parser& p, int tCount,
         TokenInstance* tParams,
         TokenInstance* tfrom )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   FalconClass* cls = ctx->currentClass();
   fassert( cls != 0 );

   TokenInstance* ti = 0;
   
   bool isObject = false;

   // a symbol class?
   if( cls->name().size() != 0 )
   {
      // we could also get the objectness of the class context in ctx...
      isObject = cls->name().getCharAt(0) == '%';

      // register as a global class...
      bool ok = ctx->onOpenClass( cls, isObject );
      if( ! ok )
      {
          p.addError( e_already_def,  p.currentSource(), cls->sr().line(), cls->sr().chr() );
           // however, go on with class creation
          if ( sp.interactive() )
          {
             // unless interactive...
             ctx->dropContext();
             p.popState();

             delete cls;
             p.simplify( tCount );
             return;
          }
          // make this anonymous.
          cls->name("");
      }
   }
   else {
      // ... but we have an expression value
      ti = TokenInstance::alloc( p.currentLine(), p.currentLexer()->character(), sp.Expr);
      Expression* expr = new ExprValue( FALCON_GC_STORE( cls->handler(), cls ),
               cls->sr().line(), cls->sr().chr() );
      ti->setValue( expr, treestep_deletor );
   }

   // Some parameters to take care of?
   if( tParams != 0 )
   {
      NameList* list = static_cast<NameList*>( tParams->asData() );
      Function* func = cls->makeConstructor();
      SymbolMap& symtab = func->parameters();

      for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
      {
         symtab.insert( *it );
      }
   }

   // some from clause to take care of?
   if( tfrom != 0 )
   {
      ExprParentship* flist = static_cast<ExprParentship*>( tfrom->detachValue() );
      cls->setParentship( flist );
      
      // check symbols in the parentship list
      for( int i = 0; i < flist->arity(); ++i ) {
         ExprInherit* inh = static_cast<ExprInherit*>( flist->nth(i) );
         // ask the owner the required symbol -- we're fine with locals.
         ctx->accessSymbol( inh->symbol()->name() );
      }
   }

   // remove this stuff from the stack
   p.simplify(tCount, ti);
   // remove the ClassStart state
   p.popState();

   p.pushState( "ClassBody" );
}

static void internal_apply_class_statement( Parser& p, bool isObject )
{
   // << T_class/T_Object << T_Name
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   p.getNextToken(); // T_class
   TokenInstance* tname=p.getNextToken();

   // Are we already in a function?
   if( ctx->currentFunc() != 0 || ctx->currentClass() != 0 || ctx->currentStmt() != 0 )
   {
      p.addError( e_toplevel_class, p.currentSource(), tname->line(), tname->chr() );
      p.consumeUpTo(p.T_EOL);
      return;
   }

   // check if the symbol is free -- defining an unique symbol
   String name;
   if( isObject ) {
      name = "%";
   }
   name += *tname->asString();

   // Create the class.
   FalconClass* cls = new FalconClass( name );
   cls->sr().line(tname->line());
   cls->sr().chr(tname->chr());

   p.simplify(2);

   if( isObject ) {
      p.pushState( "ObjectStart", false );
   }
   else {
      p.pushState( "ClassStart", false );
   }
   ctx->openClass(cls, isObject);
}

void apply_class_statement( const NonTerminal&, Parser& p )
{
   internal_apply_class_statement( p, false );
}

void apply_object_statement( const NonTerminal&, Parser& p )
{
   internal_apply_class_statement( p, true );
}



void apply_static_pdecl_expr( const NonTerminal&, Parser& p )
{
   // << T_static << T_Name << T_EqSign << Expr << T_EOL;
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   // we should be in class state.
   FalconClass* cls = (FalconClass*) ctx->currentClass();
   fassert( cls != 0 );

   sp.getNextToken(); // T_Static
   TokenInstance* tname = sp.getNextToken(); // T_Name
   sp.getNextToken(); // =
   TokenInstance* texpr = sp.getNextToken();
   sp.getNextToken(); // 'EOL'

   Expression* expr = (Expression*) texpr->detachValue();
   if( expr->trait() == Expression::e_trait_value )
   {
      cls->addProperty( *tname->asString(), static_cast<ExprValue*>(expr)->item(), true );
      // we don't need the expression anymore
      delete expr;
   }
   else
   {
      delete expr;
      p.addError( e_static_const, p.currentSource(), tname->line(), tname->chr() );
   }

   // remove this stuff from the stack
   p.simplify( 5 );
}


void apply_pdecl_expr( const NonTerminal&, Parser& p )
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
   ctx->accessSymbols(expr);
   if( expr->trait() == Expression::e_trait_value )
   {
      cls->addProperty( *tname->asString(), static_cast<ExprValue*>(expr)->item() );
      // we don't need the expression anymore
      delete expr;
   }
   else
   {
      cls->addProperty( *tname->asString(), expr );
   }

   // remove this stuff from the stack
   p.simplify( 4 );
}


//=================================================================
// Init clause
//

void apply_init_expr( const NonTerminal&, Parser& p )
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
      SynFunc* ctor = cls->makeConstructor();            
      ctx->openFunc( static_cast<SynFunc*>(ctor) );
      p.simplify( 2 );
      p.pushState("Main");
   }
}

//=================================================================
// From clause
//

void apply_FromClause_next( const NonTerminal&, Parser& p  )
{
   // << FromClause << T_Comma << FromEntry
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tInhList = p.getNextToken(); // FromClause
   p.getNextToken(); // T_Comma
   TokenInstance* tInh = p.getNextToken(); // FromEntry

   // keep the list, but discard the comma and the entry.
   ExprParentship* inhList = static_cast<ExprParentship*>( tInhList->detachValue() );
   inhList->append( static_cast<ExprInherit*>(tInh->detachValue()) );

   TokenInstance* tiNew = TokenInstance::alloc(tInhList->line(), tInhList->chr(), sp.FromClause );
   tiNew->setValue( inhList, &treestep_deletor );
   p.simplify(3, tiNew );
}


void apply_FromClause_first( const NonTerminal&, Parser& p )
{
   // << FromEntry
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tInh = p.getNextToken(); // ListExpr

   TokenInstance* tlist = TokenInstance::alloc(tInh->line(), tInh->chr(), sp.FromClause );
   ExprParentship* ep = new ExprParentship( tInh->line(), tInh->chr() );   
   ep->append( static_cast<ExprInherit*>(tInh->detachValue()) );
   tlist->setValue( ep, &treestep_deletor );

   p.simplify(1, tlist);
}


void apply_FromClause_entry_with_expr( const NonTerminal&, Parser& p )
{
   // << T_Name << T_Openpar << ListExpr << T_Closepar );
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tname = p.getNextToken(); // T_Name
   p.getNextToken();
   TokenInstance* tlistExpr = p.getNextToken(); // ListExpr

   //TODO Save the token location
   ExprInherit* ei = new ExprInherit( *tname->asString(), tname->line(), tname->chr());
   
   List* list=static_cast<List*>(tlistExpr->asData());
   for(List::const_iterator it=list->begin(),end=list->end();it!=end;++it)
   {
      ei->append( *it );
   }
   list->clear();
   
   // eventually add a requirement.
   

   TokenInstance* tInh = TokenInstance::alloc(tname->line(), tname->chr(), sp.FromEntry );
   tInh->setValue( ei, &treestep_deletor );

   p.simplify( 4, tInh );
}


void apply_FromClause_entry( const NonTerminal&, Parser& p )
{
   // << T_Name );
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tname = p.getNextToken(); // T_Name

   TokenInstance* tInh = TokenInstance::alloc(tname->line(), tname->chr(), sp.FromEntry );
   ExprInherit* ei = new ExprInherit( *tname->asString(), tname->line(), tname->chr());
   tInh->setValue( ei, treestep_deletor );
   p.simplify( 1, tInh );
}


//======================================================================
// Anon classes.
//

void apply_expr_class( const NonTerminal&, Parser& p)
{
   // T_class
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   TokenInstance* tcls=p.getNextToken();

   // let the class be anonymous
   FalconClass* cls = new FalconClass( "" );
   cls->sr().line(tcls->line());
   cls->sr().chr(tcls->chr());

   p.simplify(1);
   p.pushState("ClassStart", false);

   ctx->openClass(cls, false);
}

void apply_class_from( const NonTerminal&, Parser& p )
{
   //<< T_from << FromClause << T_EOL
   p.getNextToken(); // T_from
   TokenInstance* tfrom=p.getNextToken();

   make_class(p, 3, 0, tfrom );
   
}


void apply_class( const NonTerminal&, Parser& p )
{
   // << T_EOL
   make_class(p, 1, 0, 0 );
}


void apply_class_p_from( const NonTerminal&, Parser& p )
{
   // << T_Openpar << ListSymbol << T_Closepar << T_from << FromClause << T_EOL
   p.getNextToken(); // T_Openpar
   TokenInstance* tparams = p.getNextToken();
   p.getNextToken(); // T_Closepar
   p.getNextToken(); // T_from
   TokenInstance* tfrom = p.getNextToken();

   make_class(p, 6, tparams, tfrom );
}


void apply_class_p( const NonTerminal&, Parser& p )
{
   // << T_Openpar << ListSymbol << T_Closepar  << T_EOL
   p.getNextToken(); // T_Openpar
   TokenInstance* tparams = p.getNextToken();
   
   make_class(p, 4, tparams, 0 );

}

}

/* end of parser_class.cpp */
