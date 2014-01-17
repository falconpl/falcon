/*
   FALCON - The Falcon Programming Language.
   FILE: apache_reply.cpp

   Web Oriented Programming Interface

   Apache-specific reply object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Feb 2010 10:43:35 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/autocstring.h>
#include <apr_strings.h>

#include "apache_reply.h"
#include "apache_output.h"
#include "apache_stream.h"

using namespace Falcon;

namespace { // do not export this stuff

class ApacheCommitHandler: public WOPI::Reply::CommitHandler
{
public:
   ApacheCommitHandler( request_rec* req ):
      m_request(req)
   {
   }

   virtual ~ApacheCommitHandler() {}

   //! Invoked when the commit operation is about to begin.
    virtual void startCommit( WOPI::Reply* reply )
    {
       m_request->status = reply->status();
       Falcon::AutoCString csline( reply->reason() );
       m_request->status_line = apr_pstrdup ( m_request->pool, csline.c_str() );
    }

   //! Invoked to finalize an header
   virtual void commitHeader( WOPI::Reply*, const String& hname, const String& hvalue )
   {
      Falcon::AutoCString cvalue( hvalue );

      if ( hname.compareIgnoreCase( "Content-Type" ) == 0 )
      {
         m_request->content_type = apr_pstrdup ( m_request->pool, cvalue.c_str() );
      }
      // else {
      Falcon::AutoCString cname( hname );
      apr_table_add( m_request->headers_out, cname.c_str(), cvalue.c_str() );
      //}
   }

   //! Invoked when all the headers are generated.
   virtual void endCommit( WOPI::Reply* )
   {
      // nothing to do
   }

private:
   request_rec* m_request;
};

}

//===================================================================================
// Apache specific WOPI reply
//

ApacheReply::ApacheReply( Falcon::WOPI::ModuleWopi* mod, request_rec* req ):
   WOPI::Reply( mod ),
   m_request( req )
{
   setCommitHandler( new ApacheCommitHandler(req) );
}


ApacheReply::~ApacheReply()
{
}

/* end of apache_reply.cpp */
