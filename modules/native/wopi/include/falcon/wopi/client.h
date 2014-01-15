/*
   FALCON - The Falcon Programming Language.
   FILE: client.h

   Web Oriented Programming Interface.

   Abstract representation of a client in the WOPI system.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 15 Jan 2014 12:36:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_WOPI_CLIENT_H_
#define _FALCON_WOPI_CLIENT_H_

#include <falcon/wopi/request.h>
#include <falcon/wopi/reply.h>
#include <falcon/wopi/mem_sm.h>

namespace Falcon {
namespace WOPI {

/** Abstract representation of a client in the WOPI system.
 *
 * The base representation of a WOPI client is a set of a request,
 * a reply and a Falcon Stream reading from the request and writing
 * to the reply.
 *
 * The client owns the request and reply object, unless detach()
 * is invoked; In that case, the ownership passess onto the caller.
 */
class Client
{
public:
   Client( Request* req, Reply* reply, Stream* stream );
   virtual ~Client();

   void sendData( const String& sReply );
   virtual void sendData( const void* data, uint32 size );

   Stream* stream() const { return m_stream; }
   Request* request() const { return m_request; }
   Reply* reply() const { return m_reply; }

   /** Abide ownership of request and reply objects */
   void detach() { m_bOwnReqRep = false; }

   virtual void close();
   /** Read everything from the request. */
   virtual void consumeRequest();

   /** Declare the client completely served. */
   void complete( bool mode = true) { m_bComplete = mode; }

   bool isComplete() const { return m_bComplete; }

protected:

   Request* m_request;
   Reply* m_reply;
   Stream* m_stream;
   bool m_bComplete;
   bool m_bOwnReqRep;
};

}
}

#endif /* _FALCON_WOPI_CLIENT_H_ */

/* client.h */
