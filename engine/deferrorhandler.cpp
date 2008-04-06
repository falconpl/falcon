/*
   FALCON - The Falcon Programming Language
   FILE: deferrorhandler.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ago 25 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#include <falcon/string.h>
#include <falcon/deferrorhandler.h>
#include <falcon/stream.h>

namespace Falcon {

void DefaultErrorHandler::handleError( Error *err )
{
   String temp;
   err->toString( temp );
   m_stream->writeString( temp );
   m_stream->flush();
}

DefaultErrorHandler::~DefaultErrorHandler()
{
   if ( m_streamOwner )
      delete m_stream;
}

void DefaultErrorHandler::setStream( Stream *stream, bool owner )
{
   if ( m_streamOwner )
      delete m_stream;

   m_stream = stream;
   m_streamOwner = owner;
}

}


/* end of deferrorhandler.cpp */
