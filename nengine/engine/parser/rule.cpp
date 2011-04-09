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

#include <falcon/parser/rule.h>
#include <falcon/string.h>

#include <vector>

namespace Falcon {
namespace Parser {

class Rule::Private
{
private:
   friend class Rule;
   friend class Rule::Maker;

   typedef std::vector<Token*> TokenVector;
   TokenVector m_vTokens;
   
};

Rule::Maker::Maker( const String& name, Apply app ):
   m_name(name),
   m_apply(app)
{
   _p = new Rule::Private;
}

Rule::Maker::~Maker()
{
   delete _p;
}

Rule::Maker& Rule::Maker::t( Token& t )
{
   _p->m_vTokens.push_back( &t );
}


Rule::Rule( const String& name, Apply app )
{
   _p = new Rule::Private;
}


Rule::Rule( const Maker& m ):
   m_name( m.m_name ),
   m_apply( m.m_apply )
{
   _p = m._p;
   m._p = 0;
}


Rule::~Rule()
{
   delete _p;
}

Rule& Rule::t( Token& t )
{
   _p->m_vTokens.push_back( &t );
}


Rule::t_matchType Rule::match( const Parser& p )
{
   return t_tooShort;
}


}
}

/* end of parser/rule.cpp */
