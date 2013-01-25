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

#include <falcon/parser/rule.h>
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


bool classdecl_errhand(const NonTerminal&, Parser& p)
{
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* ti = p.getNextToken();
   
   // already detected?
   if( p.lastErrorLine() != ti->line() )
   {
      p.addError( e_syn_import, p.currentSource(), ti->line(), ti->chr() );
   }
   
   // remove the whole line
   p.consumeUpTo( p.T_EOL );
   p.clearFrames();
   
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
   }
      
   // we need to create a discardable anonymous class if we're a module.   
   return true;
}

using namespace Parsing;

static void make_class( Parser& p, int tCount,
         TokenInstance* tname,
         TokenInstance* tParams,
         TokenInstance* tfrom )
{
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   FalconClass* cls = 0;
   Variable* symclass = 0;
   TokenInstance* ti = 0;
   
   // a symbol class?
   if ( tname != 0 )
   {
      // Are we already in a function?
      if( ctx->currentFunc() != 0 || ctx->currentClass() != 0 || ctx->currentStmt() != 0 )
      {
         p.addError( e_toplevel_class,  p.currentSource(), tname->line(), tname->chr() );
         p.simplify( tCount  );
         return;
      }

      // check if the symbol is free -- defining an unique symbol
      cls = new FalconClass( *tname->asString() );
      symclass = ctx->onOpenClass( cls, false );
      if( symclass == 0 )
      {
          p.addError( e_already_def,  p.currentSource(), tname->line(), tname->chr(), 0,
                    String("at line ").N(symclass->declaredAt()) );
           // however, go on with class creation
          if ( sp.interactive() )
          {
             // unless interactive...
             delete cls;
             p.simplify( tCount );
             return;
          }
          // make this anonymous.
          cls->name("");
      }
   }
   else
   {
      // we don't have a symbol...
      cls = new FalconClass( "" );
      symclass = 0;
      
      // ... but we have an expression value
      ti = TokenInstance::alloc( p.currentLine(), p.currentLexer()->character(), sp.Expr);
      Expression* expr = new ExprValue( FALCON_GC_STORE( cls->handler(), cls ), p.currentLine(), p.currentLexer()->character() );
      ti->setValue( expr, expr_deletor );
   }

   // Some parameters to take care of?
   if( tParams != 0 )
   {
      NameList* list = static_cast<NameList*>( tParams->asData() );
      Function* func = cls->makeConstructor();
      VarMap& symtab = func->variables();

      for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
      {
         symtab.addParam( *it );
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
         // ask the owner the required symbol.
         Variable* symBaseClass = ctx->onGlobalAccessed( inh->name() );
         // if it's 0, we need a requirement; and we shall also add an externa symbol.
         if( symBaseClass->type() == Variable::e_nt_extern )
         {
            // then just add the requirement
            Requirement* req = inh->makeRequirement( cls );
            ctx->onRequirement( req );
         }
         else {
            // if it's defined and not a class, we're in trouble
            const Item* value = ctx->getVariableValue( inh->name(), symBaseClass );
            fassert( value != 0 );

            if( value == 0 || ! value->isClass() )
            {
               p.addError( e_inv_inherit, p.currentSource(), ti->line(), ti->chr() );
               p.simplify(tCount);
               return;
            }
            // cool, we can configure the inheritance.
            inh->base( static_cast<Class*>(value->asInst()) );
         }
      }
   }

   // remove this stuff from the stack
   p.simplify(tCount, ti);
   // remove the ClassStart state?
   if ( tname == 0 )
   {
      p.popState();
   }
   
   ctx->openClass(cls, false );

   p.pushState( "ClassBody" );
}


void apply_class( const Rule&, Parser& p )
{
   // << T_class << T_Name << T_EOL
   p.getNextToken(); // T_class
   TokenInstance* tname=p.getNextToken();

   make_class(p, 3, tname, 0, 0 );
}


void apply_class_p( const Rule&, Parser& p )
{
   // << T_class << T_Name << T_Openpar << ListSymbol << T_Closepar << T_EOL
   p.getNextToken(); // T_class
   TokenInstance* tname = p.getNextToken(); // T_Name
   p.getNextToken(); // T_Openpar
   TokenInstance* tparams = p.getNextToken(); // ListSymbol

   make_class( p, 6, tname, tparams, 0 );
}


void apply_class_from( const Rule&, Parser& p )
{
   // << T_class << T_Name << T_from << FromClause << T_EOL
   p.getNextToken(); // T_class
   TokenInstance* tname=p.getNextToken(); // T_Name
   (void) p.getNextToken();
   TokenInstance* tfrom=p.getNextToken(); // FromClause

   make_class(p, 5, tname, 0, tfrom );
}


void apply_class_p_from( const Rule&, Parser& p )
{
   // << T_class << T_Name << T_Openpar << ListSymbol << T_Closepar << T_from << FromClause << T_EOL
   p.getNextToken(); // T_class
   TokenInstance* tname = p.getNextToken(); // T_Name
   p.getNextToken(); // T_Openpar
   TokenInstance* tparams = p.getNextToken(); // ListSymbol
   p.getNextToken(); // T_Closepar
   p.getNextToken();
   TokenInstance* tfrom=p.getNextToken(); // FromClause

   make_class( p, 8, tname, tparams, tfrom );
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
      SynFunc* ctor = cls->makeConstructor();            
      ctx->openFunc( static_cast<SynFunc*>(ctor) );
      p.simplify( 2 );
      p.pushState("Main");
   }
}

//=================================================================
// From clause
//

void apply_FromClause_next( const Rule&, Parser& p  )
{
   // << FromClause << T_Comma << FromEntry
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tInhList = p.getNextToken(); // FromClause
   p.getNextToken(); // T_Comma
   TokenInstance* tInh = p.getNextToken(); // FromEntry

   // keep the list, but discard the comma and the entry.
   ExprParentship* inhList = static_cast<ExprParentship*>( tInhList->detachValue() );
   inhList->add( static_cast<ExprInherit*>(tInh->detachValue()) );

   TokenInstance* tiNew = TokenInstance::alloc(tInhList->line(), tInhList->chr(), sp.FromClause );
   tiNew->setValue( inhList, &expr_deletor );
   p.simplify(3, tiNew );
}


void apply_FromClause_first( const Rule&, Parser& p )
{
   // << FromEntry
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tInh = p.getNextToken(); // ListExpr

   TokenInstance* tlist = TokenInstance::alloc(tInh->line(), tInh->chr(), sp.FromClause );
   ExprParentship* ep = new ExprParentship( tInh->line(), tInh->chr() );   
   ep->add( static_cast<ExprInherit*>(tInh->detachValue()) );
   tlist->setValue( ep, &expr_deletor );

   p.simplify(1, tlist);
}


void apply_FromClause_entry_with_expr( const Rule&, Parser& p )
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
      ei->add( *it );
   }
   list->clear();
   
   // eventually add a requirement.
   

   TokenInstance* tInh = TokenInstance::alloc(tname->line(), tname->chr(), sp.FromEntry );
   tInh->setValue( ei, &expr_deletor );

   p.simplify( 4, tInh );
}


void apply_FromClause_entry( const Rule&, Parser& p )
{
   // << T_Name );
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tname = p.getNextToken(); // T_Name

   TokenInstance* tInh = TokenInstance::alloc(tname->line(), tname->chr(), sp.FromEntry );
   ExprInherit* ei = new ExprInherit( *tname->asString(), tname->line(), tname->chr());
   tInh->setValue( ei, expr_deletor );
   p.simplify( 1, tInh );
}


//======================================================================
// Anon classes.
//

void apply_expr_class(const Rule&, Parser& p)
{
   // T_class
   p.simplify(1);
   p.pushState( "ClassStart", false );
}


void apply_anonclass_from( const Rule&, Parser& p )
{
   //<< T_from << FromClause << T_EOL
   p.getNextToken(); // T_from
   TokenInstance* tfrom=p.getNextToken();

   make_class(p, 3, 0, 0, tfrom );
   
}


void apply_anonclass( const Rule&, Parser& p )
{
   // << T_EOL
   make_class(p, 1, 0, 0, 0 );
}


void apply_anonclass_p_from( const Rule&, Parser& p )
{
   // << T_Openpar << ListSymbol << T_Closepar << T_from << FromClause << T_EOL
   p.getNextToken(); // T_Openpar
   TokenInstance* tparams = p.getNextToken();
   p.getNextToken(); // T_Closepar
   p.getNextToken(); // T_from
   TokenInstance* tfrom = p.getNextToken();

   make_class(p, 3, 0, tparams, tfrom );
}


void apply_anonclass_p( const Rule&, Parser& p )
{
   // << T_Openpar << ListSymbol << T_Closepar  << T_EOL
   p.getNextToken(); // T_Openpar
   TokenInstance* tparams = p.getNextToken();
   
   make_class(p, 3, 0, tparams, 0 );

}

}

/* end of parser_class.cpp */
