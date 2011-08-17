/*
   FALCON - The Falcon Programming Language.
   FILE: rosstream.h

   Definition of read only string stream.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab ago 19 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
   virtual int32 write( const void *buffer, int32 size );
   virtual int32 write( const String &source );
   virtual int32 writeAvailable( int32 msecs, const Falcon::Sys::SystemData* );
   virtual bool truncate( int64 pos=-1 );
   virtual ROStringStream *clone() const;
};

}

#endif

/* end of rosstream.h */
