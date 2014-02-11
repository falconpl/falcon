/*
   FALCON - The Falcon Programming Language.
   FILE: parser/token.cpp

   Token for the parser subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Apr 2011 17:16:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/parser/token.cpp"

#include <falcon/parser/token.h>
#include <falcon/parser/tokeninstance.h>

namespace Falcon {
namespace Parsing {

Token::Token(const String& name, int prio, bool bRightAssoc):
   m_bNonTerminal(false),
   m_bRightAssoc( bRightAssoc ),
   m_prio(prio),
   m_name(name)
{
}

Token::Token()
{
}

void Token::name( const String& n )
{
    m_name = n;
}


Token::~Token()
{
}

TokenInstance* Token::makeInstance( int line, int chr )
{
   TokenInstance* ti = TokenInstance::alloc( line, chr, *this);
   return ti;
}

TokenInstance* Token::makeInstance( int line, int chr, void* data, deletor d )
{
   TokenInstance* ti = TokenInstance::alloc( line, chr, *this);
   ti->setValue( data, d );
   return ti;
}

TokenInstance* Token::makeInstance( int line, int chr, int32 v )
{
   TokenInstance* ti = TokenInstance::alloc( line, chr, *this);
   ti->setValue( v );
   return ti;
}

TokenInstance* Token::makeInstance( int line, int chr, uint32 v )
{
   TokenInstance* ti = TokenInstance::alloc( line, chr, *this);
   ti->setValue( v );
   return ti;
}

TokenInstance* Token::makeInstance( int line, int chr, int64 v )
{
   TokenInstance* ti = TokenInstance::alloc(  line, chr, *this);
   ti->setValue( v );
   return ti;
}

TokenInstance* Token::makeInstance( int line, int chr, numeric v )
{
   TokenInstance* ti = TokenInstance::alloc( line, chr, *this);
   ti->setValue( v );
   return ti;
}

TokenInstance* Token::makeInstance( int line, int chr, bool v )
{
   TokenInstance* ti = TokenInstance::alloc( line, chr, *this);
   ti->setValue( v );
   return ti;
}

TokenInstance* Token::makeInstance( int line, int chr, const String& v )
{
   TokenInstance* ti = TokenInstance::alloc( line, chr, *this);
   ti->setValue( v );
   return ti;
}

}
}

/* end of parser/token.cpp */
