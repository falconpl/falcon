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

#include <falcon/parser/token.h>
#include <falcon/parser/tokeninstance.h>

namespace Falcon {
namespace Parser {

#define HASH_SEED 0xF3DE3EA3

Token::Token(uint32 nID, const String& name ):
   m_name( name ),
   m_nID(nID)
{}

Token::Token(const String& name):
   m_name(name)
{
   m_nID = simpleHash( name );
}

uint32 Token::simpleHash( const String& v )
{
   uint32 h = HASH_SEED;
   
   for( length_t i = 0; i < v.length(); ++i )
   {
      char_t chr = v.getCharAt(i);
      if( chr > 0xFFFF )
      {
         h += chr;
      }
      else if( chr > 0xFF )
      {
         h += chr << (16*(i%2));
      }
      else
      {
         h += chr << (8*(i%3));
      }
   }

   return h;
}

Token::~Token()
{
}

TokenInstance* Token::makeInstance( void* data, deletor d )
{
   TokenInstance* ti = new TokenInstance(*this);
   ti->setValue( data, d );
   return ti;
}

TokenInstance* Token::makeInstance( int32 v )
{
   TokenInstance* ti = new TokenInstance(*this);
   ti->setValue( v );
   return ti;
}

TokenInstance* Token::makeInstance( uint32 v )
{
   TokenInstance* ti = new TokenInstance(*this);
   ti->setValue( v );
   return ti;
}

TokenInstance* Token::makeInstance( int64 v )
{
   TokenInstance* ti = new TokenInstance(*this);
   ti->setValue( v );
   return ti;
}

TokenInstance* Token::makeInstance( numeric v )
{
   TokenInstance* ti = new TokenInstance(*this);
   ti->setValue( v );
   return ti;
}

TokenInstance* Token::makeInstance( bool v )
{
   TokenInstance* ti = new TokenInstance(*this);
   ti->setValue( v );
   return ti;
}

TokenInstance* Token::makeInstance( const String& v )
{
   TokenInstance* ti = new TokenInstance(*this);
   ti->setValue( v );
   return ti;
}

}
}

/* end of parser/token.cpp */

