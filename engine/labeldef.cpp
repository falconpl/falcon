/*
   FALCON - The Falcon Programming Language.
   FILE: labeldef.cpp
   $Id: labeldef.cpp,v 1.2 2006/10/15 20:21:50 gian Exp $

   Definition for assembly oriented labels.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab ago 27 2005
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
   Definition for assembly oriented labels.
*/

#include <fasm/labeldef.h>
#include <falcon/string.h>
#include <falcon/common.h>
#include <falcon/stream.h>

namespace Falcon
{

LabelDef::LabelDef( const String &name ):
   m_position( FASM_UNDEFINED_LABEL )
{
   m_name.bufferize( name );
}

LabelDef::~LabelDef()
{
}


void LabelDef::write( Stream *os )
{
   uint32 pos = endianInt32( m_position );

   if ( ! defined() ) {
      m_forwards.pushBack( (void *) os->tell() );
   }

   os->write( &pos, sizeof( pos ) );
}


void LabelDef::defineNow( Stream *os )
{
   if ( defined() )
      return;

   m_position = static_cast< uint32 >( os->tell() );

   if ( ! m_forwards.empty() )
   {
      uint32 pos = endianInt32( m_position );
      ListElement *fw = m_forwards.begin();

      while( os->good() && fw != 0 )
      {
         os->seekBegin( fw->iData() );
         os->write( &pos, sizeof( pos ) );
         fw = fw->next();
      }

      // just to spare memory
      m_forwards.clear();
      os->seekBegin( m_position );
   }
}


}

/* end of labeldef.cpp */
