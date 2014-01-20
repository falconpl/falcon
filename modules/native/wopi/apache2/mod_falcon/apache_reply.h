/*
   FALCON - The Falcon Programming Language.
   FILE: apache_reply.h

   Web Oriented Programming Interface

   Object encapsulating reply.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Feb 2010 10:43:35 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_APACHE_REPLY
#define FALCON_APACHE_REPLY

#include <falcon/wopi/reply.h>

#include <httpd.h>
#include <http_request.h>

/** Apache-specific reply manager */
class ApacheReply: public Falcon::WOPI::Reply
{
public:
   ApacheReply( Falcon::WOPI::ModuleWopi* mod, request_rec* req );
   virtual ~ApacheReply();

private:
   request_rec* m_request;
};

#endif

/* end of apache_reply.h */
