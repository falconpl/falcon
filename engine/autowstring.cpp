/*
   FALCON - The Falcon Programming Language
   FILE: autowstring.cpp

   Utility to convert falcon items and strings into Wide C Strings.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab ago 4 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Utility to convert falcon items and strings into Wide C Strings.
*/

#include <falcon/autowstring.h>
#include <falcon/item.h>
#include <falcon/vm.h>

namespace Falcon {

AutoWString::AutoWString():
   m_len(0)
{
   m_pData = m_buffer;
   m_buffer[0] = 0;
}

AutoWString::AutoWString( const String &str ):
   m_pData(0)
{
   set( str );
}


AutoWString::~AutoWString()
{
   if ( m_pData != 0 && m_pData != m_buffer )
      delete[] m_pData;
}

void AutoWString::set( const Falcon::String &str )
{
   if ( m_pData != 0 && m_pData != m_buffer )
   {
      delete[] m_pData ;
   }

   if ( (m_len = str.toWideString( m_buffer, AutoWString_BUF_SPACE ) ) != String::npos )
   {
      m_pData = m_buffer;
      return;
   }

   Falcon::uint32 len = str.length() + 1;
   m_pData = new wchar_t[ len ];
   m_len = str.toWideString( m_pData, len*2 );
}

}


/* end of autowstring.cpp */
