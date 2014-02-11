/*
   FALCON - The Falcon Programming Language.
   FILE: parser/rule.cpp

   Token for the parser subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Apr 2011 21:17:45 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/parser/rule.cpp"

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>
#include "./parser_private.h"
#include <falcon/fassert.h>
#include <falcon/parser/token.h>
#include <falcon/parser/tokeninstance.h>
#include <falcon/parser/nonterminal.h>
#include <falcon/string.h>

#include <falcon/trace.h>

#include <vector>


namespace Falcon {
namespace Parsing {

class Rule::Private
{
private:
   friend class Rule;

   typedef std::vector<Token*> TokenVector;
   TokenVector m_vTokens;

};

//=================================================================
//

Rule::Rule():
   m_bGreedy( false ),
   m_bRecursive( false )
{
   _p = new Rule::Private;
}

Rule::Rule( const String& name, Apply app ):
   m_name(name),
   m_apply(app),
   m_bGreedy( false ),
   m_bRecursive( false )
{
   _p = new Rule::Private;
}

Rule::~Rule()
{
   delete _p;
}

Rule& Rule::t( Token& t )
{
   _p->m_vTokens.push_back( &t );
   if( t.isNT() )
   {
      /*
      NonTerminal& nt = *static_cast<NonTerminal*>(&t);
      // check for greedness: the rule is greedy if it has ends with a recursive token.
      if( nt.isRecursive() )
      {
         m_bGreedy = true;
      }
      else
      {
         m_bGreedy = false;
      }
      */
   }
   else
   {
      // rules terminating with a non-terminal can never bee greedy
      m_bGreedy = false;
   }

   return *this;
}


void Rule::parent( NonTerminal& nt )
{
   m_parent = &nt;

   m_bRecursive = false;
   Private::TokenVector::iterator riter = _p->m_vTokens.begin();
   Private::TokenVector::iterator riter_end = _p->m_vTokens.end();
   while( riter != riter_end )
   {
      if ( nt.id() ==  (*riter)->id() )
      {
         m_bRecursive = true;
         return;
      }
      ++riter;
   }

}

int Rule::arity() const
{
   return _p->m_vTokens.size();
}

void Rule::apply( Parser& parser ) const
{
   parser.resetNextToken();
   m_apply(*this, parser);
}

Token* Rule::getTokenAt( uint32 pos ) const
{
   if( pos >= _p->m_vTokens.size() )
   {
      return 0;
   }

   return _p->m_vTokens[pos];
}

bool Rule::match( Parser& parser, bool bIncremental, bool bContinue ) const
{
   /*
   TRACE2( "Rule::match(%s) -- %s/%s", m_name.c_ize(),
            bIncremental ? "incremental" : "full", bContinue ? "cont" : "anew" );
*/
   Parser::Private* pp = parser._p;
   size_t begin = pp->m_pframes->back().m_nStackDepth;
   size_t ppos = bIncremental ? pp->m_tokenStack->size() - 1 : pp->m_pframes->back().m_nStackDepth;
   size_t ppos_end = pp->m_tokenStack->size();

   if( _p->m_vTokens.empty() )
   {
      if( ppos + 1 >= pp->m_tokenStack->size() )
      {
         TRACE1( "Rule::match(%s) -- always matching when at end", m_name.c_ize() );
         return true;
      }
      else
      {
         TRACE1( "Rule::match(%s) -- always failing when in the middle", m_name.c_ize() );
         return false;
      }
   }


   Private::TokenVector::iterator riter = _p->m_vTokens.begin();
   if( bIncremental )
   {
      // we can't possibly match on this one.
      if( ppos - begin >  _p->m_vTokens.size() )
      {
         return false;
      }

      riter += ppos - begin;
   }
   Private::TokenVector::iterator riter_end = _p->m_vTokens.end();

   // descendable position
   int dpos = -1;
   NonTerminal* descendable = 0;

   // check if there is a match by scanning the current token stack in the parser
   // -- and matching it against our tokens.
   while ( riter != riter_end && ppos < ppos_end )
   {
      Token* curTok = *riter;
      const Token* stackToken = &(*pp->m_tokenStack)[ppos]->token();

      if( curTok->isNT() )
      {
         if( (ppos> begin || curTok->id() != m_parent->id()) )
         {
            TRACE3( "Rule::match(%s) -- descendable '%s' found at %d",
               m_name.c_ize(), curTok->name().c_ize(), (int)ppos );
            dpos = ppos;
            descendable = static_cast<NonTerminal*>(curTok);
         }
      }

      if( curTok->id() != stackToken->id() || bContinue )
      {
         // actually, descendable should always be != 0 when dpos != -1, but just in case...
         if( dpos != -1 && descendable != 0 )
         {
            TRACE1( "Rule::match(%s) -- actually descending '%s' found at %d%s",
               m_name.c_ize(), descendable->name().c_ize(), dpos, dpos > (int) begin ? " (adding stack)": "" );

            parser.pushParseFrame( descendable, dpos );
            return false; //descendable->findPaths(parser);
         }

         // match failed
         //TRACE2( "Rule::match(%s) -- failed at %d", m_name.c_ize(), (int)ppos );
         return false;
      }
      else
      {
         // only terminal tokens can have priority, but we don't care.
      }

      ++ riter;
      ++ ppos;
   }

   TRACE1( "Rule::match(%s) -- matched (mode %s) at %d", m_name.c_ize(),
            bIncremental ? "incremental" : "full", (int) ppos );
   return true;
}

}
}

/* end of parser/rule.cpp */
