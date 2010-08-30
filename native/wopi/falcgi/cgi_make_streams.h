/*
   FALCON - The Falcon Programming Language.
   FILE: cgi_make_streams.h

   Falcon CGI program driver - common declaration for stream provider.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Feb 2010 17:35:14 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef CGI_MAKE_STREAMS_H_
#define CGI_MAKE_STREAMS_H_

#include <falcon/stream.h>

Falcon::Stream* makeOutputStream();
Falcon::Stream* makeInputStream();
Falcon::Stream* makeErrorStream();

#endif /* CGI_MAKE_STREAMS_H_ */

/* end of cgi_make_streams.h */
