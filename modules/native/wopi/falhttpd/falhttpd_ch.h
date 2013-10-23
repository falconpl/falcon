/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_reply.h

   Micro HTTPD server providing Falcon scripts on the web.

   Reply object specific for the Falcon micro HTTPD server.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 06 Mar 2010 21:31:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALHTTPD_REPLY_H_
#define FALHTTPD_REPLY_H_

#include "falhttpd.h"
#include <falcon/wopi/reply.h>

namespace Falcon {

class Falhttpd_CommitHandler: public WOPI::Reply::CommitHandler
{
public:
   Falhttpd_CommitHandler();
   ~FalhttpdReply();

   virtual void startCommit( WOPI::Reply* reply, Stream* tgt );
   virtual void commitHeader( WOPI::Reply* reply, Stream* tgt, const String& hname, const String& hvalue );
   virtual void endCommit( WOPI::Reply* reply, Stream* tgt);

private:
   Falcon::String m_headers;
};
}

#endif

/* falhttpd_reply.h */
