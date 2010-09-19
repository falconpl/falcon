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
#include "apache_output.h"

/** Apache-specific reply manager */
class ApacheReply: public Falcon::WOPI::Reply
{
public:

   ApacheReply( const Falcon::CoreClass* base );
   virtual ~ApacheReply();

   void init( request_rec* r, ApacheOutput* aout );

   static Falcon::CoreObject* factory( const Falcon::CoreClass* cls, void* ud, bool bDeser );

protected:
   virtual void startCommit();
   virtual Falcon::Stream* makeOutputStream();
   virtual void commitHeader( const Falcon::String& name, const Falcon::String& value );
   virtual void endCommit();
private:
   request_rec* m_request;
   ApacheOutput* m_aout;
};

#endif

/* end of apache_reply.h */
