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

AutoWString::AutoWString( const Item &itm ):
   m_pData(0)
{
   set( itm );
}

AutoWString::~AutoWString()
{
   if ( m_pData != 0 && m_pData != m_buffer )
      memFree( m_pData );
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




void AutoWString::set( const Falcon::String &str )
{
   if ( m_pData != 0 && m_pData != m_buffer )
        memFree( m_pData );

   if ( (m_len = str.toWideString( m_buffer, AutoWString_BUF_SPACE ) ) != String::npos )
   {
      m_pData = m_buffer;
      return;
   }

   Falcon::uint32 len = str.length() * 2 + 2;
   m_pData = (wchar_t *) memAlloc( len );
   m_len = str.toWideString( m_pData, len );
}


void AutoWString::set( const Falcon::Item &itm )
{
   if ( m_pData != 0 && m_pData != m_buffer )
      memFree( m_pData );

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


void AutoWString::set( Falcon::VMachine *vm, const Falcon::Item &itm )
{
   if ( m_pData != 0 && m_pData != m_buffer )
      memFree( m_pData );

   init_vm_and_format( vm, itm, "" );
}


void AutoWString::set( Falcon::VMachine *vm, const Falcon::Item &itm, const Falcon::String &fmt )
{
   if ( m_pData != 0 && m_pData != m_buffer )
         memFree( m_pData );

   init_vm_and_format( vm, itm, fmt );
}

}


/* end of autowstring.cpp */
