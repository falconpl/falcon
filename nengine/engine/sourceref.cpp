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
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/common.h>

namespace Falcon {

void SourceRef::serialize( DataWriter *s ) const
{
   s->write( m_line );
   s->write( m_char );
}

void SourceRef::deserialize( DataReader *s )
{
   s->read( m_line );
   s->read( m_char );
}

}
/* end of sourceref.cpp */
