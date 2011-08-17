/*
   FALCON - The Falcon Programming Language
   FILE: autoucsstring.cpp

   Utility to convert falcon items and strings into UCS-2 Strings.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 12 Sep 2010 12:53:18 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Utility to convert falcon items and strings into UCS-2 Strings.
*/

#include <falcon/autoucsstring.h>
#include <falcon/item.h>
#include <falcon/vm.h>

namespace Falcon {

AutoUCSString::AutoUCSString():
   m_len(0)
{
   m_pData = m_buffer;
   m_buffer[0] = 0;
}

AutoUCSString::AutoUCSString( const String &str, uint16 defChar ):
   m_pData(0)
{
   set( str, defChar );
}

AutoUCSString::~AutoUCSString()
{
   if ( m_pData != 0 && m_pData != m_buffer )
      memFree( m_pData );
}


void AutoUCSString::set( const Falcon::String &str, uint16 defChar )
{
   if ( m_pData != 0 && m_pData != m_buffer )
        memFree( m_pData );

   m_len = str.length();
   if ( m_len + 1 <= AutoUCSString_BUF_SPACE )
   {
      m_pData = m_buffer;
   }
   else
   {
      Falcon::uint32 size = (m_len + 1) * sizeof(uint16);
      m_pData = (uint16 *) memAlloc( size );
   }

   for( uint32 i = 0; i < m_len; ++i )
   {
      uint32 chr = str.getCharAt(i);
      m_pData[i] = chr <= 0xFFFF ? chr : defChar;
   }

   m_pData[m_len] = 0;
}

}


/* end of autoucsstring.cpp */
