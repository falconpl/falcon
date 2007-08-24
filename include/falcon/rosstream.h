/*
   FALCON - The Falcon Programming Language.
   FILE: rosstream.h
   $Id: rosstream.h,v 1.3 2007/08/18 20:10:18 jonnymind Exp $

   Definition of read only string stream.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab ago 19 2006
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
   Definition of read only string stream.
*/

#ifndef flc_rosstream_H
#define flc_rosstream_H

#include <falcon/stringstream.h>

namespace Falcon {

/** Read only string stream for fast operations.
   The base class StringStream copies strings used as source data.

   When the string stream is just bound to read from the string
   as if it were a file, i.e. with encoders, this is not desirable.

   This implementation takes the buffer of the incoming string
   as stream bytes source, and never alter or destroys the
   content of the given buffer. The original string must stay
   available and ummodified for the whole life of the read only
   string stream, or bad things will happen.

   This stringstream can be constructed also with a static char *
   source, so that it is possible to use statically written code
   as source of streams.
*/

class FALCON_DYN_CLASS ROStringStream: public StringStream
{
public:
   ROStringStream( const String &source );
   ROStringStream( const char *source, int size = -1 );
   ROStringStream( const ROStringStream &other );

   virtual ~ROStringStream() { close(); }

   virtual bool close();
   virtual int32 write( const byte *buffer, int32 size );
   virtual int32 write( const String &source );
   virtual int32 writeAvailable( int32 msecs );
   virtual bool truncate( int64 pos=-1 );
   virtual UserData *clone();
};

}

#endif

/* end of rosstream.h */
