/*
   FALCON - The Falcon Programming Language.
   FILE: apache_request.h

   Falcon module for Apache 2
   Apache specific WOPI request
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Feb 2010 15:36:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef APACHE_REQUEST_H
#define APACHE_REQUEST_H

#include <falcon/wopi/request.h>

#include <httpd.h>
#include <http_request.h>

#define BOUNDARY_SIZE   128

class ApacheRequest: public Falcon::WOPI::Request
{
public:

   ApacheRequest( Falcon::WOPI::ModuleWopi*, request_rec* req );
   virtual ~ApacheRequest();

   //! parse the header part.
   /** \note this inserts Falcon GC relevant objects in the GC,
    *  if invoked from outside the VM, wrap in gc-disabled zone.
    *  (Also, the Request object should be already locked/reachable from GC)
    */
   virtual void parseHeader( Falcon::Stream* input );

   request_rec* apacheRequest() const { return m_request; }

private:

   request_rec* m_request;
};

#endif

/* end of apache_requeset.h */
