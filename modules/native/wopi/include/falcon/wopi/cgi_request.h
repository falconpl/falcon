/*
   FALCON - The Falcon Programming Language.
   FILE: cgi_request.h

   Falcon CGI program driver - cgi-based request specialization.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 21 Feb 2010 12:19:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_WOPI_CGI_REQUEST_H_
#define _FALCON_WOPI_CGI_REQUEST_H_

#include <falcon/wopi/request.h>

namespace Falcon {
namespace WOPI {

class CGIRequest: public Request
{
public:
   CGIRequest( ModuleWopi* host );
   virtual ~CGIRequest();

   void init( Stream* input );
   void PostInitPrepare( Falcon::VMachine* vmowner );

   virtual void postInit();
   Falcon::int64 m_cration_time;

private:

   static void handleEnvStr( const Falcon::String& key, const Falcon::String& value, void *data );

   void addHeaderFromEnv( const Falcon::String& key, const Falcon::String& value );
   void readPostFields( Falcon::Stream* input );
   void readMultipartFields( Falcon::Stream* input );
   bool readSinglePart( const Falcon::String& sBoundary, Falcon::Stream* input  );

   int m_post_length;
   Falcon::String m_post_type;

   Falcon::VMachine* m_vmOwner;
};


}
}


#endif /* CGI_REQUEST_H_ */

/* end of cgi_request.h */
