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
#include <falcon/vm.h>

namespace Falcon {

AutoCString::AutoCString():
   m_len(0)
{
   m_pData = m_buffer;
   m_buffer[3] = '\0';
}
AutoCString::AutoCString( const String &str ):
   m_pData(0)
{
   set(str);
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
   if ( m_pData != 0 && m_pData != m_buffer )
free( m_pData );
}


void AutoCString::set( const Falcon::String &str )
{
   // remove m_pData
   if( m_pData != 0 && m_pData != m_buffer )
free( m_pData );

   if ( (m_len = str.toCString( m_buffer+3, AutoCString_BUF_SPACE-3 ) ) != String::npos )
   {
     m_pData = m_buffer;
     return;
   }

   Falcon::uint32 len = str.length() * 4 + 4;
   m_pData = (char *) malloc( len+3 );
   m_len = str.toCString( m_pData+3, len );
}

}


/* end of autocstring.cpp */
