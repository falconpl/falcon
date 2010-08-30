/*
   FALCON - The Falcon Programming Language.
   FILE: mod_falcon_errhand.h

   Falcon module for Apache 2

   Apache aware error handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 30 Apr 2008 18:18:08 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef APACHE_ERRHAND_H
#define APACHE_ERRHAND_H

#include <falcon/error.h>

#include "apache_output.h"

/** Error handler specialized for the apache module.
   This error handler knows how to handle output.

   The scripts may change the
*/
class ApacheErrorHandler
{
   int m_notifyMode;
   ApacheOutput *m_output;

   // Todo: add support for encoding.
public:
   ApacheErrorHandler( int nmode, ApacheOutput *aout ):
      m_notifyMode( nmode ),
      m_output( aout )
   {}

   void notifyMode( int nmode ) { m_notifyMode = nmode; }
   int notifyMode() const { return m_notifyMode; }

   virtual void handleError( Falcon::Error *error );
};

#endif
