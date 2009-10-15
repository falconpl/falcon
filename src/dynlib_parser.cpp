/*
   The Falcon Programming Language
   FILE: dynlib_mod.cpp

   Direct dynamic library interface for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 13 Oct 2009 23:17:09 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: The Falcon Committee

   See the LICENSE file distributed with this package for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/fassert.h>
#include <falcon/tokenizer.h>

#include "dynlib_parser.h"

namespace Falcon
{

BaseCType::BaseCType( e_integral_type id, const String& repr, int size ):
   m_id(id),
   m_size( size ),
   m_normal_repr( repr )
{
}

BaseCType::BaseCType( e_integral_type id, const char *repr, int size ):
   m_id(id),
   m_size( size ),
   m_normal_repr( repr )
{
}

BaseCType ctypes[] =
{
   BaseCType( BaseCType::e_void, "void", 0 ),
   BaseCType( BaseCType::e_char, "char", sizeof(char) ),
   BaseCType( BaseCType::e_unsigned_char, "unsigned char", sizeof(char) ),
   BaseCType( BaseCType::e_signed_char, "signed char", sizeof(char) ),
   // ... then the modern version
   BaseCType( BaseCType::e_short, "short", sizeof(short) ),
   BaseCType( BaseCType::e_unsigned_short, "unsigned short", sizeof(short) ),
   BaseCType( BaseCType::e_int, "int", sizeof(int) ),
   BaseCType( BaseCType::e_unsigned_int, "unsigned int", sizeof(int) ),
   BaseCType( BaseCType::e_long, "long", sizeof(long) ),
   BaseCType( BaseCType::e_unsigned_long, "unsigned long", sizeof(long) ),
   BaseCType( BaseCType::e_long_long, "long long", sizeof(int64) ),
   BaseCType( BaseCType::e_unsigned_long_long, "unsigned long long", sizeof(int64) ),
   BaseCType( BaseCType::e_float, "float", sizeof(float) ),
   BaseCType( BaseCType::e_double, "double", sizeof(double) ),
   BaseCType( BaseCType::e_long_double, "long double", sizeof(long double) ),

   // tagged types
   BaseCType( BaseCType::e_struct, "struct *", sizeof(void*) ),
   BaseCType( BaseCType::e_union, "union *", sizeof(void*) ),
   BaseCType( BaseCType::e_enum, "enum *", sizeof(void*) ),

   // some aliases
   BaseCType( BaseCType::e_int, "BOOL", sizeof(int) ),
   BaseCType( BaseCType::e_long_long, "__int64", sizeof(int64) ),
   BaseCType( BaseCType::e_unsigned_long_long, "unsigned __int64", sizeof(int64) ),
   BaseCType( BaseCType::e_unsigned_short, "WORD", 2 ),
   BaseCType( BaseCType::e_unsigned_int, "DWORD", 4 ),
   BaseCType( BaseCType::e_unsigned_long, "HANDLE", sizeof(long) ),

   // Finally, varadic params
   BaseCType( BaseCType::e_varpar, "...", sizeof(void*) )

};


CType::CType( BaseCType* ct, int pointers, int subs, bool isFunc ):
   m_ctype(ct),
   m_pointers(pointers),
   m_subscript(subs),
   m_isFuncPtr(isFunc)
{}


CType::CType( BaseCType* ct, const String &tag, int pointers, int subs, bool isFunc ):
   m_ctype(ct),
   m_tag(tag),
   m_pointers(pointers),
   m_subscript(subs),
   m_isFuncPtr(isFunc)
{
   fassert( ct->m_id == BaseCType::e_struct || ct->m_id == BaseCType::e_union || ct->m_id == BaseCType::e_enum );
}


CType::CType( const CType& other ):
   m_ctype( other.m_ctype ),
   m_tag(other.m_tag),
   m_pointers(other.m_pointers),
   m_subscript(other.m_subscript),
   m_isFuncPtr(other.m_isFuncPtr),
   m_funcParams(other.m_funcParams)
{
}

CType::~CType()
{}

//=======================================================
//

Parameter::Parameter( const CType& t, const char* name ):
   m_type(t),
   m_name(name),
   m_next(0)
{
}

//=======================================================
//

ParamList::ParamList():
   m_head(0),
   m_tail(0),
   m_size(0)
   {}

ParamList::ParamList( const ParamList& other ):
   m_head(0),
   m_tail(0),
   m_size(0)
{
   Parameter* p = other.m_head;
   while (p != 0 )
   {
      add( new Parameter(*p) );
      p = p->m_next;
   }
}

ParamList::~ParamList()
{
   Parameter* p = m_head;
   while ( p != 0 )
   {
      Parameter* old = p;
      p = p->m_next;
      delete old;
   }
}

void ParamList::add(Parameter* p)
{
   if ( m_head == 0 )
   {
      m_head = m_tail = p;
   }
   else
   {
      m_tail->m_next = p;
      m_tail = p;
   }
   m_size++;
   p->m_next = 0; // just to be on the bright side.
}


//===================================================
//
//===================================================

FunctionDef2::FunctionDef2( const FunctionDef2& other ):
   m_definition( other.m_definition ),
   m_name( other.m_name ),
   m_params( other.m_params )
{
   if ( other.m_return != 0 )
      m_return = new Parameter( *other.m_return );
   else
      m_return = 0;
}

FunctionDef2::~FunctionDef2()
{
   delete m_return;
}

bool FunctionDef2::parse( const String& definition )
{
   enum {
      e_start,
      e_spec,
      e_type,
      e_tag,
      e_unsigned,
      e_signed,
      e_varadic,
      e_name
   } state;

   // do we have a function name?
   state = e_start;
   bool bFuncName = false;

   BaseCType::e_integral_type type;
   BaseCType::e_integral_type spec;

   Tokenizer t( TokenizerParams().wsIsToken().returnSep(), "(,);[]", definition );

   while( t.hasCurrent() )
   {
      const String& token = t.token();

      switch( state )
      {
      case e_start:
         if( token == "void" )
         {
            type = BaseCType::e_void;
            state = e_type;
         }
         else if( token == "int" )
         {
            type = BaseCType::e_int;
            state = e_type;
         }
         else if( token == "char" )
         {
            type = BaseCType::e_char;
            state = e_type;
         }
         else if( token == "float" )
         {
            type = BaseCType::e_float;
            state = e_type;
         }
         else if( token == "double" )
         {
            type = BaseCType::e_double;
            state = e_type;
         }
         else if( token == "BOOL" )
         {
            type = BaseCType::e_int;
            state = e_type;
         }
         else if( token == "WORD" )
         {
            type = BaseCType::e_unsigned_short;
            state = e_type;
         }
         else if( token == "DWORD" )
         {
            type = BaseCType::e_unsigned_int;
            state = e_type;
         }
         else if( token == "HANDLE" )
         {
            type = BaseCType::e_unsigned_long;
            state = e_type;
         }
         else if( token == "struct" )
         {
            type = BaseCType::e_double;
            state = e_tag;
         }
         else if( token == "union" )
         {
            type = BaseCType::e_union;
            state = e_tag;
         }
         else if( token == "enum" )
         {
            type = BaseCType::e_enum;
            state = e_tag;
         }
         else if( token == "long" )
         {
            spec = BaseCType::e_long;
            state = e_spec;
         }
         else if( token == "short" )
         {
            spec = BaseCType::e_short;
            state = e_spec;
         }
         else if( token == "unsigned" )
         {
               state = e_unsigned;
         }
         else if( token == "signed" )
         {
               state = e_signed;
         }
         else if( token == "..." )
         {
            if( ! bFuncName )
               return false;

            state = e_varadic;
         }
            return false;
         break;

      case e_spec:
         if( token == "long" )
         {
            if ( spec == BaseCType::e_long )
               spec = BaseCType::e_long_long;
            else if ( spec == BaseCType::e_unsigned_long )
               spec = BaseCType::e_unsigned_long_long;
            else
               return false;
            // and stay in this state.
         }
         else if ( token == "int" )
         {
            type = spec;
            state = e_type;
         }
         else if( token == "double" )
         {
            if( spec == BaseCType::e_long )
            {
               spec = BaseCType::e_long_double;
               state = e_type;
            }
            else
               return false;
         }
         else if( token == "short" || token == "char" || token == "float" ||
            token == "struct" || token == "union" || token == "enum" ||
            token == "void" || token == "..." )
         {
            return false;
         }
         else {
            // the token is the parameter/function name.
            current = new Parameter( spec, name );
            state = e_name;
            if ( bFuncName )
               m_parameters.add( current );
            else
               m_name = name;
               m_return = current;

      }
   }

}


}
