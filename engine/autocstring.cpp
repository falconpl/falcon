/*
   FALCON - The Falcon Programming Language
   FILE: autocstring.cpp
   $Id: autocstring.cpp,v 1.3 2007/08/18 13:23:53 jonnymind Exp $

   Utility to convert falcon items and strings into C Strings.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab ago 4 2007
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
   if ( (m_len = str->toCString( m_buffer, AutoCString_BUF_SPACE ) ) >= 0 )
   {
      m_pData = m_buffer;
      return;
   }

   Falcon::uint32 len = str->length() * 4 + 4;
   m_pData = (char *) memAlloc( len );
   m_len = str->toCString( m_pData, len );
}


AutoCString::AutoCString( const String &str ):
   m_pData(0)
{
   if ( (m_len = str.toCString( m_buffer, AutoCString_BUF_SPACE ) ) >= 0)
   {
      m_pData = m_buffer;
      return;
   }

   Falcon::uint32 len = str.length() * 4 + 4;
   m_pData = (char *) memAlloc( len );
   m_len = str.toCString( m_pData, len );
}


void AutoCString::init_vm_and_format( VMachine *vm, const Item &itm, const String &fmt )
{
   String *str;
   String tempStr;

   str = &tempStr;
   vm->itemToString( tempStr, &itm, fmt );
   if( vm->hadError() )
   {
      m_pData = m_buffer;
      // as 255 is not a valid UTF-8 character at the beginning of a string,
      // we can use it to determine if the string is valid.
      m_buffer[0] = (char) 255;
      return;
   }

   if ( (m_len = str->toCString( m_buffer, AutoCString_BUF_SPACE ) ) >= 0 )
   {
      m_pData = m_buffer;
      return;
   }

   Falcon::uint32 len = str->length() * 4 + 4;
   m_pData = (char *) memAlloc( len );
   m_len = str->toCString( m_pData, len );
}


AutoCString::~AutoCString()
{
   if ( m_pData != m_buffer )
      memFree( m_pData );
}

}


/* end of autocstring.cpp */
