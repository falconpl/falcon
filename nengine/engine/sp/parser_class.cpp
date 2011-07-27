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
#include <falcon/error.h>
#include <falcon/exprvalue.h>
#include <falcon/globalsymbol.h>
#include <falcon/falconclass.h>
#include <falcon/synfunc.h>
#include <falcon/localsymbol.h>
#include <falcon/inheritance.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/sp/parser_class.h>

#include "private_types.h"

#include <list>

namespace Falcon {

typedef std::list<Inheritance* > InhList;

void inh_list_deletor(void* data)
{
   InhList* expr = static_cast<InhList*>(data);
   InhList::iterator iter = expr->begin();
   while( iter != expr->end() )
   {
      delete *iter;
      ++iter;
   }
   delete expr;
}


void inh_deletor(void* data)
{
   delete static_cast<Inheritance*>(data);
}


using namespace Parsing;

static void make_class( Parser& p, int tCount,
         TokenInstance* tname,
         TokenInstance* tParams,
         TokenInstance* tfrom )
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
      SymbolTable& symtab = cls->makeConstructor()->symbols();

      for(NameList::const_iterator it=list->begin(),end=list->end();it!=end;++it)
      {
         symtab.addLocal( *it );
      }
      cls->makeConstructor()->paramCount( symtab.localCount() );
   }

   // some from clause to take care of?
   if( tfrom != 0 )
   {
      InhList* flist = static_cast<InhList*>( tfrom->asData() );
      InhList::iterator iter = flist->begin();
      while( iter != flist->end() )
      {
         Inheritance* inh = *iter;
         cls->addParent( inh );
         ctx->onInheritance( inh );
         ++iter;
      }

      // preserve the inheritances, but discard the list.
      flist->clear();
   }


   // remove this stuff from the stack
   p.simplify( tCount );

   ctx->openClass(cls, false, symclass);

   // time to check the symbols.
   ctx->checkSymbols();

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
      SynFunc* init = new SynFunc("init");
      // but.. we don't really know if the init funciton has been created
      if( (init = static_cast<SynFunc*>(cls->init())) == 0 )
      {
         init = new SynFunc("init");
         cls->setInit( init );
      }

      // copy the symbols of the constructor in the function
      SynFunc* constructor = cls->makeConstructor();
      for(int pCount = 0; pCount < constructor->symbols().localCount(); ++pCount )
      {
         init->symbols().addLocal( constructor->symbols().getLocal( pCount )->clone() );
      }
      
      // the user can add a non-syntree function as init,
      // ... but we won't be here compiling it.
      ctx->openFunc( static_cast<SynFunc*>(cls->init()) );
      p.simplify( 2 );
      p.pushState("Main");
   }
}


void apply_FromClause_next( const Rule&, Parser& p  )
{
   // << FromClause << T_Comma << FromEntry
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tInhList = p.getNextToken(); // FromClause
   p.getNextToken(); // T_Comma
   TokenInstance* tInh = p.getNextToken(); // FromEntry

   // keep the list, but discard the comma and the entry.
   InhList* inhList = static_cast<InhList*>( tInhList->detachValue() );
   inhList->push_back( static_cast<Inheritance*>(tInh->detachValue()) );

   TokenInstance* tiNew = new TokenInstance(tInhList->line(), tInhList->chr(), sp.FromClause );
   tiNew->setValue( inhList, &inh_list_deletor );
   p.simplify(3, tiNew );
}


void apply_FromClause_first( const Rule&, Parser& p )
{
   // << FromEntry
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tInh = p.getNextToken(); // ListExpr

   TokenInstance* tlist = new TokenInstance(tInh->line(), tInh->chr(), sp.FromClause );
   InhList* inh_list = new InhList;
   inh_list->push_back( static_cast<Inheritance*>(tInh->detachValue()) );
   tlist->setValue( inh_list, &inh_list_deletor );

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
   Inheritance* inh = new Inheritance(*tname->asString());
   
   List* list=static_cast<List*>(tlistExpr->asData());
   for(List::const_iterator it=list->begin(),end=list->end();it!=end;++it)
   {
      inh->addParameter( *it );
   }
   list->clear();

   TokenInstance* tInh = new TokenInstance(tname->line(), tname->chr(), sp.FromEntry );
   tInh->setValue( inh, &inh_deletor );

   p.simplify( 4, tInh );
}


void apply_FromClause_entry( const Rule&, Parser& p )
{
   // << T_Name );
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* tname = p.getNextToken(); // T_Name

   TokenInstance* tInh = new TokenInstance(tname->line(), tname->chr(), sp.FromEntry );
   tInh->setValue( new Inheritance(*tname->asString()), &inh_deletor );
   p.simplify( 1, tInh );
}

}

/* end of parser_class.cpp */
