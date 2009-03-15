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

AutoWString::AutoWString( const Item &itm )
{
   String *str;
   String tempStr;

   if ( itm.isString() )
   {
      str = itm.asString();
   }
   else {
      str = &tempStr;
      itm.toString( tempStr );
   }

   if ( (m_len = str->toWideString( m_buffer, AutoWString_BUF_SPACE ) ) != String::npos )
   {
      m_pData = m_buffer;
      return;
   }

   Falcon::uint32 len = str->length() * 2 + 2;
   m_pData = (wchar_t *) memAlloc( len );
   m_len = str->toWideString( m_pData, len );
}


AutoWString::AutoWString( const String &str ):
   m_pData(0)
{
   if ( (m_len = str.toWideString( m_buffer, AutoWString_BUF_SPACE ) ) != String::npos )
   {
      m_pData = m_buffer;
      return;
   }

   Falcon::uint32 len = str.length() * 2 + 2;
   m_pData = (wchar_t *) memAlloc( len );
   m_len = str.toWideString( m_pData, len );
}


void AutoWString::init_vm_and_format( VMachine *vm, const Item &itm, const String &fmt )
{
   String *str;
   String tempStr;

   str = &tempStr;
   vm->itemToString( tempStr, &itm, fmt );
   
   if ( ( m_len = str->toWideString( m_buffer, AutoWString_BUF_SPACE ) ) != String::npos )
   {
      m_pData = m_buffer;
      return;
   }

   Falcon::uint32 len = str->length() * 2 + 2;
   m_pData = (wchar_t *) memAlloc( len );
   m_len = str->toWideString( m_pData, len );
}


AutoWString::~AutoWString()
{
   if ( m_pData != m_buffer )
      memFree( m_pData );
}

}


/* end of autowstring.cpp */
