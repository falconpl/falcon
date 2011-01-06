/*
   FALCON - The Falcon Programming Language.
   FILE: sourceref.cpp

   Syntactic tree item definitions -- position in a source file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/sourceref.h>
#include <falcon/stream.h>

namespace Falcon {

void SourceRef::serialize( Stream *s )
{
   int32 line;

   line = endianInt32(m_line);
   s->write( &line, sizeof(line) );

   line = endianInt32(m_char);
   s->write( &line, sizeof(line) );
}

void SourceRef::deserialize( Stream *s )
{
   int32 line;
   s->read(&line, 4 );
   m_line = endianInt32(line);

   s->read(&line, 4 );
   m_char = endianInt32(line);
}

}
/* end of sourceref.cpp */
