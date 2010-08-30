/*
   FALCON - The Falcon Programming Language.
   FILE: cgi_reply.h

   Falcon CGI program driver - cgi-based reply specialization.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 21 Feb 2010 12:19:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef CGI_REPLY_H_
#define CGI_REPLY_H_

#include <falcon/wopi/reply.h>
#include <falcon/stream.h>

class CGIReply: public Falcon::WOPI::Reply
{
public:
   CGIReply( const Falcon::CoreClass* cls );
   virtual ~CGIReply();

   void init();

   static Falcon::CoreObject* factory( const Falcon::CoreClass* cls, void* ud, bool bDeser );

protected:
   virtual void startCommit();
   virtual Falcon::Stream* makeOutputStream();
   virtual void commitHeader( const Falcon::String& name, const Falcon::String& value );
   virtual void endCommit();

private:
   Falcon::Stream* m_output;
};

#endif /* CGI_REPLY_H_ */

/* end of cgi_reply.h */
