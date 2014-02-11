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
#include <falcon/parser/parser.h>

#include <falcon/psteps/exprself.h>
#include <falcon/psteps/exprfself.h>
#include <falcon/psteps/exprsym.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/psteps/expristring.h>
#include <falcon/psteps/exprinit.h>
#include <falcon/psteps/exprautoclone.h>
#include <falcon/stdhandlers.h>

#include <falcon/error.h>

#include <falcon/symbol.h>
#include "../re2/re2/re2.h"

namespace Falcon {

using namespace Parsing;

void apply_Atom_Int ( const NonTerminal&, Parser& p )
{
   // << (r_Atom_Int << "Atom_Int" << apply_Atom_Int << T_Int )

   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();

   ti->token( sp.Atom );
   ti->setValue( new ExprValue((int64) ti->asInteger(), ti->line(), ti->chr()), treestep_deletor );
}


void apply_Atom_Float ( const NonTerminal&, Parser& p )
{
   // << (r_Atom_Float << "Atom_Float" << apply_Atom_Float << T_Float )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   ti->token( sp.Atom );
   ti->setValue( new ExprValue(ti->asNumeric(), ti->line(), ti->chr()), treestep_deletor );
}


void apply_Atom_Name ( const NonTerminal&, Parser& p )
{
   static Engine* inst = Engine::instance();
   
   // << (r_Atom_Name << "Atom_Name" << apply_Atom_Name << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   const Item* builtin = inst->getBuiltin( *ti->asString() );
   Expression* sym; 

   if( builtin )
   {
      sym = new ExprValue( *builtin, ti->line(), ti->chr() );
   }
   else
   {
      //TODO: check for globalized variables instead of using isGlobalContext
      const String& name = *ti->asString();
      Symbol* s = Engine::getSymbol( name );
      sym = new ExprSymbol( s, ti->line(), ti->chr() );
      // exprsymbol doesn't incref s.
   }

   ti->token( sp.Atom );
   ti->setValue( sym, treestep_deletor );
}



void apply_Atom_Pure_Name ( const NonTerminal&, Parser& p )
{
   // << "Atom_Pure_Name" << apply_Atom_Pure_Name << T_Tilde << T_Name )
   SourceParser& sp = static_cast<SourceParser&>(p);
   TokenInstance* tilde = p.getNextToken(); // T_Tilde;
   TokenInstance* ti = p.getNextToken();

   const String& name = *ti->asString();
   Symbol* sym = Engine::getSymbol( name );
   ExprSymbol* esym = new ExprSymbol( sym, ti->line(), ti->chr() );
   esym->setPure(true);
   // exprsymbol doesn't incref s.

   p.trim(1); // remove T_Name...
   tilde->token( sp.Atom );     // change T_Tilde...
   tilde->setValue( esym, treestep_deletor );
}


void apply_Atom_String ( const NonTerminal&, Parser& p )
{
   // << (r_Atom_String << "Atom_String" << apply_Atom_String << T_String )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* ti = p.getNextToken();

   // get the string and it's class, to generate a static UserValue
   String* s = ti->detachString();
   s->setImmutable(true);
   // The exprvalue is made so that it will gc lock the string.
   Expression* res = new ExprValue( FALCON_GC_HANDLE(s), ti->line(), ti->chr() );
   ti->token( sp.Atom );
   ti->setValue( res, treestep_deletor );
}


void apply_Atom_RString ( const NonTerminal&, Parser& p )
{
   static Class* sc = Engine::handlers()->reClass();

   // << (r_Atom_String << "Atom_String" << apply_Atom_String << T_String )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* ti = p.getNextToken();

   // get the string and it's class, to generate a static UserValue
   Expression* res;

   String* s = ti->detachString();

   //==============================================
   // scan for options
   re2::RE2::Options opts;
   uint32 count = 1;
   uint32 slen = s->length();
   while( count < slen && count <= 7 )
   {
      uint32 realLen = slen-count;
      uint32 chr = s->getCharAt(realLen);

      // do we have options?
      if( chr == 0 )
      {
         --count;
         while( count > 0 )
         {
            chr = s->getCharAt(slen - count);
            switch( chr )
            {
            case 'i': opts.set_case_sensitive(false); break;
            case 'n': opts.set_never_nl(true); break;
            case 'l': opts.set_longest_match(true); break;
            case 'o': opts.set_one_line(true); break;
            default:
               p.addError( e_regex_def,
                                 p.currentSource(), ti->line(), ti->chr(), 0, "Unknown option" );
               break;
            }

            if( p.hasErrors() )
            {
               break;
            }

            --count;
         }

         s->size( realLen * s->manipulator()->charSize() );
         break;
      }

      count++;
   }

   //==============================================
   // Generate the regex
   //
   re2::RE2* regex = new re2::RE2(*s, opts);

   if( ! regex->ok() )
   {
      String errDesc;
      errDesc.fromUTF8(regex->error().c_str());
      String temp;
      temp.fromUTF8( regex->error_arg().c_str() );
      errDesc += " at ";
      errDesc += temp;
      p.addError( e_regex_def,
                  p.currentSource(), ti->line(), ti->chr(), 0, errDesc );
      delete regex;
      res = new ExprValue( Item(), ti->line(), ti->chr() );
   }
   else {
      // The exprvalue is made so that it will gc lock the string.
      res = new ExprValue( FALCON_GC_STORE(sc, regex), ti->line(), ti->chr() );
   }

   ti->token( sp.Atom );
   ti->setValue( res, treestep_deletor );
}


void apply_Atom_MString ( const NonTerminal&, Parser& p )
{
   // << (r_Atom_String << "Atom_String" << apply_Atom_String << T_String )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* ti = p.getNextToken();

   // get the string and it's class, to generate a static UserValue
   String* s = ti->detachString();
   Expression* res = new ExprAutoClone( s->handler(), s, ti->line(), ti->chr() );

   ti->token( sp.Atom );
   ti->setValue( res, treestep_deletor );
}

void apply_Atom_IString ( const NonTerminal&, Parser& p )
{
   // << (r_Atom_String << "Atom_String" << apply_Atom_String << T_String )
   SourceParser& sp = static_cast<SourceParser&>(p);
   //ParserContext* ctx = static_cast<ParserContext*>(p.context());
   TokenInstance* ti = p.getNextToken();

   // get the string and it's class, to generate a static UserValue
   String* s = ti->detachString();
   // The exprvalue is made so that it will gc lock the string.
   Expression* res = new ExprIString(*s, ti->line(), ti->chr() );
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());
   ctx->onIString(*s);

   delete s;
   ti->token( sp.Atom );
   ti->setValue( res, treestep_deletor );
}

void apply_Atom_False ( const NonTerminal&, Parser& p )
{
   // << (r_Atom_Nil << "Atom_Nil" << apply_Atom_Nil << T_Nil )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   ti->token( sp.Atom );
   ti->setValue( new ExprValue(Item(false), ti->line(), ti->chr() ), treestep_deletor );
}


void apply_Atom_True ( const NonTerminal&, Parser& p )
{
   // << (r_Atom_Nil << "Atom_Nil" << apply_Atom_Nil << T_Nil )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   ti->token( sp.Atom );
   Item val;
   val.setBoolean(true);
   ti->setValue( new ExprValue(val, ti->line(), ti->chr()), treestep_deletor );
}

void apply_Atom_Self ( const NonTerminal&, Parser& p )
{
   // << (r_Atom_Nil << "Atom_Self" << apply_Atom_Delf << T_self )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   ti->token(sp.Atom);
   ti->setValue( new ExprSelf, treestep_deletor );
}

void apply_Atom_FSelf ( const NonTerminal&, Parser& p )
{
   // << (r_Atom_Nil << "Atom_Self" << apply_Atom_Delf << T_fself )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   ti->token(sp.Atom);
   ti->setValue( new ExprFSelf, treestep_deletor );
}

void apply_Atom_Init ( const NonTerminal&, Parser& p )
{
   // << (r_Atom_Nil << "Atom_Self" << apply_Atom_Delf << T_init )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   ti->token(sp.Atom);
   ti->setValue( new ExprInit, treestep_deletor );
}

void apply_Atom_Nil ( const NonTerminal&, Parser& p )
{
   // << (r_Atom_Nil << "Atom_Nil" << apply_Atom_Nil << T_Nil )
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   ti->token( sp.Atom );
   ti->setValue( new ExprValue(Item(), ti->line(), ti->chr()), treestep_deletor );
}

void apply_expr_atom( const NonTerminal&, Parser& p )
{
   SourceParser& sp = static_cast<SourceParser&>(p);

   TokenInstance* ti = p.getNextToken();
   Expression* expr = (Expression*) ti->detachValue();
   
   ti->token( sp.Expr );
   ti->setValue( expr, treestep_deletor );
}

}

/* end of parser_atom.cpp */
