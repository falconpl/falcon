/*
   FALCON - The Falcon Programming Language.
   FILE: sstreamapi.h
   $Id: stringstream_srv.h,v 1.1.1.1 2006/10/08 15:05:40 gian Exp $

   String Stream service publisher.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mar 5 2006
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
