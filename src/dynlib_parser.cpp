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

#include "dynlib_parser.h"

namespace Falcon
{

BaseCType::BaseCType( e_integral_type id, const char *repr ):
   m_id(id),
   m_normal_repr( repr )
{
}

BaseCType ctypes[] =
{
   BaseCType( BaseCType::e_void, "void" ),
   BaseCType( BaseCType::e_char, "char" ),
   BaseCType( BaseCType::e_unsigned_char, "unsigned char" ),
   BaseCType( BaseCType::e_signed_char, "signed char" ),
   // long decl (old) version before to be sure to parse it
   BaseCType( BaseCType::e_short, "short int" ),
   BaseCType( BaseCType::e_unsigned_short, "unsigned short int" ),
   // ... then the modern version
   BaseCType( BaseCType::e_short, "short" ),
   BaseCType( BaseCType::e_unsigned_short, "unsigned short" ),
   BaseCType( BaseCType::e_int, "int" ),
   BaseCType( BaseCType::e_unsigned_int, "unsigned int" ),
   BaseCType( BaseCType::e_long, "long" ),
   BaseCType( BaseCType::e_unsigned_long, "unsigned long" ),
   BaseCType( BaseCType::e_long_long, "long long" ),
   BaseCType( BaseCType::e_unsigned_long_long, "unsigned long long" ),
   BaseCType( BaseCType::e_float, "float" ),
   BaseCType( BaseCType::e_double, "double" ),
   BaseCType( BaseCType::e_long_double, "long double" ),

   // tagged types
   BaseCType( BaseCType::e_struct, "struct *" ),
   BaseCType( BaseCType::e_unioin, "union *" ),
   BaseCType( BaseCType::e_enum, "enum *" ),

   // some aliases
   BaseCType( BaseCType::e_int, "BOOL" ),
   BaseCType( BaseCType::e_long_long, "__int64" ),
   BaseCType( BaseCType::e_unsigned_long_long, "unsigned __int64" ),
   BaseCType( BaseCType::e_unsigned_short, "WORD" ),
   BaseCType( BaseCType::e_unsigned_int, "DWORD" ),
   BaseCType( BaseCType::e_unsigned_long, "HANDLE" ),


};


CType::CType( BaseCType* ct, int pointers, int subs, bool isFunc ):
   m_ctype(ct),
   m_pointers(pointers),
   m_subscript(subs),
   m_isFuncPtr(isFunc),
   m_funcParams(0)
{}


CType::CType( BaseCType* ct, const String &tag, int pointers, int subs, bool isFunc ):
   m_ctype(ct),
   m_tag(tag),
   m_pointers(pointers),
   m_subscript(subs),
   m_isFuncPtr(isFunc),
   m_funcParams(0)
{
   fassert( m_id == e_struct || m_id == e_union || m_id == e_enum );
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

}
