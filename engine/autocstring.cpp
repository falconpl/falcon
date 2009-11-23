/*
   FALCON - The Falcon Programming Language
   FILE: autocstring.cpp

   Utility to convert falcon items and strings into C Strings.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab ago 4 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Utility to convert falcon items and strings into C Strings.
*/

#include <falcon/autocstring.h>
#include <falcon/item.h>
#include <falcon/vm.h>

namespace Falcon {

AutoCString::AutoCString( const Item &itm )
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

   //if the size is large, we already know we can't try the conversion
   if ( (m_len = str->toCString( m_buffer+3, AutoCString_BUF_SPACE ) ) != String::npos )
   {
      m_pData = m_buffer;
      return;
   }

   Falcon::uint32 len = str->length() * 4 + 4;
   m_pData = (char *) memAlloc( len+3 );
   m_len = str->toCString( m_pData+3, len );
}


AutoCString::AutoCString( const String &str ):
   m_pData(0)
{
   if ( (m_len = str.toCString( m_buffer+3, AutoCString_BUF_SPACE-3 ) ) != String::npos )
   {
      m_pData = m_buffer;
      return;
   }

   Falcon::uint32 len = str.length() * 4 + 4;
   m_pData = (char *) memAlloc( len+3 );
   m_len = str.toCString( m_pData+3, len );
}


void AutoCString::init_vm_and_format( VMachine *vm, const Item &itm, const String &fmt )
{
   String *str;
   String tempStr;

   str = &tempStr;
   vm->itemToString( tempStr, &itm, fmt );
   
   if ( (m_len = str->toCString( m_buffer+3, AutoCString_BUF_SPACE-3 ) ) != String::npos )
   {
      m_pData = m_buffer;
      return;
   }

   Falcon::uint32 len = str->length() * 4 + 4;
   m_pData = (char *) memAlloc( len+3 );
   m_len = str->toCString( m_pData+3, len );
}

const char* AutoCString::bom_str()
{
   m_pData[0] = (char)0xEF;
   m_pData[1] = (char)0xBB;
   m_pData[2] = (char)0xBF;
   return m_pData;
}

AutoCString::~AutoCString()
{
   if ( m_pData != m_buffer )
      memFree( m_pData );
}

}


/* end of autocstring.cpp */
