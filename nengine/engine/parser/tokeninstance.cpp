/*
   FALCON - The Falcon Programming Language.
   FILE: parser/tokeninstance.cpp

   Actual token value as determined by the lexer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Apr 2011 16:44:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/parser/token.h>
#include <falcon/parser/tokeninstance.h>
#include <falcon/string.h>

namespace Falcon {
namespace Parser {

/** Creates a new token instance.*/
TokenInstance::TokenInstance( const Token& tok  ):
   m_token( tok ),
   m_deletor(0)
{
   m_v.v_int = 0;
}

TokenInstance::~TokenInstance()
{
   clear();
}



void TokenInstance::setValue( void* v, TokenInstance::deletor d )
{
   clear();
   m_deletor = d;
   m_v.v_voidp = v;
}

void TokenInstance::setValue( int32 v )
{
   clear();
   m_v.v_int = v;
}

void TokenInstance::setValue( int64 v )
{
   clear();
   m_v.v_int = v;
}

void TokenInstance::setValue( uint32 v )
{
   clear();
   m_v.v_int = (int64) v;
}

void TokenInstance::setValue( numeric v )
{
   clear();
   m_v.v_double = v;
}

void TokenInstance::setValue( bool b )
{
   clear();
   m_v.v_bool = b;
}

static void s_string_deletor( void* s )
{
   delete (String*) s;
}

void TokenInstance::setValue( const String& str )
{
   clear();
   m_deletor = s_string_deletor;
   m_v.v_voidp = new String( str );
}

void* TokenInstance::detachValue()
{
   void* ret = m_v.v_voidp;
   m_v.v_voidp = 0;
   m_deletor = 0;
   return ret;
}

}
}

/* end of parser/tokeninstance.cpp */



