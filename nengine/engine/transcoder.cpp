/*
   FALCON - The Falcon Programming Language.
   FILE: textdecoder.cpp

   Transcoder for text encodings
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Mar 2011 13:04:14 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/transcoder.h>

namespace Falcon {

Transcoder::Transcoder( const String &name ):
   m_name( name )
{
}

Transcoder::~Transcoder()
{
}

}

/* textdecoder.cpp */
