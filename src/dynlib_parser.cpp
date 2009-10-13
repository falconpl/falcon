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

CType::CType( const CType& other ):
   m_id(other.m_id),
   m_normal_repr(other.m_normal_repr),
   m_tag(other.m_tag),
   m_pointers(other.m_pointers),
   m_subscript(other.m_subscript),
   m_isFuncPtr(other.m_isFuncPtr),
   m_funcParams(other.m_funcParams)
{
}

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
