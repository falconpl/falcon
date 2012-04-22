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

#define SRC "engine/parser/tokeninstance.cpp"

#include <falcon/parser/token.h>
#include <falcon/parser/tokeninstance.h>
#include <falcon/string.h>
#include <falcon/pool.h>

#include <map>

#include <falcon/engine.h>

namespace Falcon {
namespace Parsing {

TokenInstance::TokenInstance( int line, int chr, const Token& tok  ):
   m_line( line ),
   m_chr( chr ),
   m_token( &tok ),
   m_deletor(0)
{
   m_v.v_int = 0;
}

TokenInstance::~TokenInstance()
{
   clear();
}


TokenInstance* TokenInstance::alloc( int line, int chr, const Token& tok )
{
   static Pool* pool = 0;
   if( pool == 0 )
   {
      pool = new Pool(100);
      Engine::instance()->addPool( pool );
   }
   
   TokenInstance* ti = static_cast<TokenInstance*>(pool->get());
   if( ti == 0 ) {
      ti = new TokenInstance(line, chr, tok);
      ti->assignToPool( pool );
   }
   else {
      ti->~TokenInstance();
      try {
         ti = new(ti) TokenInstance( line, chr, tok );
      }
      catch( ... ) {
         delete ti;
      }
   }
   return ti;
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
