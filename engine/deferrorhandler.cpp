/*
   FALCON - The Falcon Programming Language
   FILE: deferrorhandler.cpp
   $Id: deferrorhandler.cpp,v 1.6 2007/03/08 14:31:44 jonnymind Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ago 25 2006
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
