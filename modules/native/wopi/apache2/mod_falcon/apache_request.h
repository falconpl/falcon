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
#include <falcon/wopi/session_manager.h>

#include <httpd.h>
#include <http_request.h>

#define BOUNDARY_SIZE   128

class ApacheRequest: public Falcon::WOPI::CoreRequest
{
public:
   struct table_callback
   {
      request_rec* request;
      Falcon::ItemDict *headers;
      Falcon::ItemDict *cookies;
      // multipart management
      bool bIsMultiPart;
      int contentLength;
      char boundary[BOUNDARY_SIZE];
      int boundaryLen;
   };

   ApacheRequest( const Falcon::CoreClass* base );
   virtual ~ApacheRequest();

   void init( request_rec* request,
         Falcon::CoreClass* upld_c,
         Falcon::WOPI::Reply* rep,
         Falcon::WOPI::SessionManager* sm  );

   void process();

   static Falcon::CoreObject* factory( const Falcon::CoreClass* cls, void* ud, bool bDeser );

private:

   request_rec* m_request;
};

#endif

/* end of apache_requeset.h */
