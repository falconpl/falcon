/*
   FALCON - The Falcon Programming Language.
   FILE: deferrorhandler.h
   $Id: deferrorhandler.h,v 1.4 2007/03/04 17:39:02 jonnymind Exp $

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

#ifndef flc_deferrorhandler_H
#define flc_deferrorhandler_H

#include <falcon/errhand.h>

namespace Falcon {

class Stream;

/** The DEFAULT error handler.
   This class converts an error in a text and writes it on a given stream.
   This minimal sensible action is however enough to manage the vast majority
   of error handling needs, as the stream may be i.e. a string stream and
   the error status can always be detected after error handler user return.
   In example, if there is just the need to intercept the error signalation
   and i.e. write it on a window, the default error handler, provided with a
   string stream, may be used to get a complete string representation of the
   error; then the host program may print it.
*/

class FALCON_DYN_CLASS DefaultErrorHandler: public ErrorHandler
{
   Stream *m_stream;
   bool m_streamOwner;

public:

   /** Creates the default error handler setting the output stream.
      The stream must have output capabilities.
      The stream is only written with writeString() and put() calls,
      so transcoders can be used as well.

       If owner is true, the stream will be destroyed on handler
       destruction.
   */
   DefaultErrorHandler( Stream *stream, bool owner = false ):
      m_stream( stream ),
      m_streamOwner( owner )
   {}

   /** Default constructor.
      Useful to set the handler at a later moment.
   */

   DefaultErrorHandler():
      m_stream( 0 ),
      m_streamOwner( false )
   {}

   ~DefaultErrorHandler();

   virtual void handleError( Error *preformatted );

   /** Sets the output stream used by this error handler.
      \note If the previously used stream was owned by this instance,
      it is destroyed here.
   */
   void setStream( Stream *stream, bool owner = false );
};

}

#endif

/* end of deferrorhandler.h */
