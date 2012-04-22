/*
   FALCON - The Falcon Programming Language.
   FILE: parser_atom.cpp

   Parser for Falcon source files -- handler for atoms and values
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:24:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_atom.cpp"

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_atom.h>
#include <falcon/sp/parser_deletor.h>

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>

#include <falcon/psteps/exprself.h>
#include <falcon/psteps/exprsym.h>
#include <falcon/psteps/exprvalue.h>

#include <falcon/symbol.h>

namespace Falcon {

using namespace Parsing;

void apply_Atom_Int ( const Rule&, Parser& p )
{
   // << (r_Atom_Int << "Atom_Int" << apply_Atom_Int << T_Int )

   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();

   TokenInstance* ti2 = TokenInstance::alloc(ti->line(), ti->chr(), sp.Atom );
   ti2->setValue( new ExprValue((int64) ti->asInteger(), ti->line(), ti->chr()), expr_deletor );
   p.simplify(1,ti2);
}


void apply_Atom_Float ( const Rule&, Parser& p )
{
   // << (r_Atom_Float << "Atom_Float" << apply_Atom_Float << T_Float )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   TokenInstance* ti2 = TokenInstance::alloc(ti->line(), ti->chr(), sp.Atom );
   ti2->setValue( new ExprValue(ti->asNumeric(), ti->line(), ti->chr()), expr_deletor );
   p.simplify(1,ti2);
}


void apply_Atom_Name ( const Rule&, Parser& p )
{
   static Engine* inst = Engine::instance();
   
   // << (r_Atom_Name << "Atom_Name" << apply_Atom_Name << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());

   TokenInstance* ti = p.getNextToken();
   const Item* builtin = inst->getBuiltin( *ti->asString() );
   Expression* sym; 

   if( builtin )
   {
      sym = new ExprValue( *builtin, ti->line(), ti->chr() );
   }
   else
   {
      sym = ctx->addVariable(*ti->asString()); 
   }

   TokenInstance* ti2 = TokenInstance::alloc(ti->line(), ti->chr(), sp.Atom );
   ti2->setValue( sym, expr_deletor );
   p.simplify(1,ti2);
}


void apply_Atom_String ( const Rule&, Parser& p )
{
   static Class* sc = Engine::instance()->stringClass();

   // << (r_Atom_String << "Atom_String" << apply_Atom_String << T_String )
   SourceParser& sp = static_cast<SourceParser&>(p);
   ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* ti = p.getNextToken();

   TokenInstance* ti2 = TokenInstance::alloc(ti->line(), ti->chr(), sp.Atom );

   // get the string and it's class, to generate a static UserValue
   String* s = ti->detachString();
   // tell the context that we have a new string around.
   Expression* res = ctx->onStaticData( sc, s );
   res->decl( ti->line(), ti->chr() );
   ti2->setValue( res, expr_deletor );

   // remove the token in the stack.
   p.simplify(1,ti2);
}


void apply_Atom_False ( const Rule&, Parser& p )
{
   // << (r_Atom_Nil << "Atom_Nil" << apply_Atom_Nil << T_Nil )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   TokenInstance* ti2 = TokenInstance::alloc(ti->line(), ti->chr(), sp.Atom );
   ti2->setValue( new ExprValue(Item(false), ti->line(), ti->chr() ), expr_deletor );
   p.simplify(1,ti2);
}


void apply_Atom_True ( const Rule&, Parser& p )
{
   // << (r_Atom_Nil << "Atom_Nil" << apply_Atom_Nil << T_Nil )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   TokenInstance* ti2 = TokenInstance::alloc(ti->line(), ti->chr(), sp.Atom );
   ti2->setValue( new ExprValue(Item(true), ti->line(), ti->chr()), expr_deletor );
   p.simplify(1,ti2);
}

void apply_Atom_Self ( const Rule&, Parser& p )
{
   // << (r_Atom_Nil << "Atom_Self" << apply_Atom_Delf << T_Nil )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   TokenInstance* ti2 = TokenInstance::alloc(ti->line(), ti->chr(), sp.Atom );
   ti2->setValue( new ExprSelf, expr_deletor );
   p.simplify(1,ti2);
}

void apply_Atom_Continue( const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);   
   TokenInstance* ti = p.getNextToken();
   
   Item cont;
   cont.setContinue();
   ti->token( sp.Atom );
   ti->setValue( new ExprValue(cont, ti->line(), ti->chr()), expr_deletor );
}

void apply_Atom_Break ( const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);   
   TokenInstance* ti = p.getNextToken();
   
   Item b;
   b.setBreak();
   ti->token( sp.Atom );
   ti->setValue( new ExprValue(b, ti->line(), ti->chr()), expr_deletor );
}

void apply_Atom_Nil ( const Rule&, Parser& p )
{
   // << (r_Atom_Nil << "Atom_Nil" << apply_Atom_Nil << T_Nil )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   TokenInstance* ti2 = TokenInstance::alloc(ti->line(), ti->chr(), sp.Atom );
   ti2->setValue( new ExprValue(Item(), ti->line(), ti->chr()), expr_deletor );
   p.simplify(1,ti2);
}

void apply_expr_atom( const Rule&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();

   TokenInstance* ti2 = TokenInstance::alloc(ti->line(), ti->chr(), sp.Expr );
   ti2->setValue( ti->detachValue(), expr_deletor );
   p.simplify(1,ti2);
}


}

/* end of parser_atom.cpp */
