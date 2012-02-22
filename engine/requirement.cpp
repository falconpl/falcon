/*
   FALCON - The Falcon Programming Language.
   FILE: requirement.cpp

   Structure holding information about classes needed elsewhere.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 15 Aug 2011 13:58:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/requirement.cpp"

#include <falcon/requirement.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>

namespace Falcon
{

void Requirement::store( DataWriter* stream ) const
{
   stream->write( m_name );
   stream->write( m_bIsStatic );
   m_sr.serialize( stream );
}


void Requirement::restore( DataReader* stream ) const
{   
   stream->read( m_name );
   stream->read( m_bIsStatic );
   m_sr.deserialize(stream);   
}


void Requirement::flatten( ItemArray& subItems ) const
{
}


void Requirement::unflatten( ItemArray& subItems ) const
{
}

}

/* end of rrequirement.cpp */
