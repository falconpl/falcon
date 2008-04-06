/*
   FALCON - The Falcon Programming Language.
   FILE: sstreamapi.h

   String Stream service publisher.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mar 5 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   String Stream service publisher.
   Import this service ("StringStream") from RTL module if your embedding application needs
   to create Falcon::StingStreams.
*/

#ifndef flc_stringstream_srv_H
#define flc_stringstream_srv_H

#include <falcon/setup.h>
#include <falcon/service.h>
#include <falcon/stringstream.h>

namespace Falcon {

class StringStreamService: public Service
{
public:
   StringStreamService():
      Service("StringStream")
   {}

   virtual StringStream *create();
   virtual StringStream *create( int32 size );
   virtual StringStream *create( String *origin );
};

}

#endif

/* end of sstreamapi.h */
