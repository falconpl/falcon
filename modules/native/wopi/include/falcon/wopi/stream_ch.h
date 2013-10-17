/*
   FALCON - The Falcon Programming Language.
   FILE: stream_ch.h

   Falcon CGI program driver - cgi-based reply specialization.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 21 Feb 2010 12:19:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_WOPI_STREAMCOMMITHANDLER_H_
#define _FALCON_WOPI_STREAMCOMMITHANDLER_H_

#include <falcon/wopi/reply.h>
#include <falcon/stream.h>

namespace Falcon {
namespace WOPI  {

class StreamCommitHandler: public Reply::CommitHandler
{
public:
   StreamCommitHandler();
   virtual ~StreamCommitHandler();

   virtual void startCommit( Reply* reply, Stream* tgt );
   virtual void commitHeader( Reply* reply, Stream* tgt, const Falcon::String& name, const Falcon::String& value );
   virtual void endCommit( Reply* reply, Stream* tgt );
};

}
}

#endif /* _FALCON_WOPI_CGICOMMITHANDLER_H_ */

/* end of stream_ch.h */
