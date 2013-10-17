/*
   FALCON - The Falcon Programming Language.
   FILE: classrequest.h

   Falcon Web Oriented Programming Interface.

   Interface to Request object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 16 Oct 2013 12:38:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_WOPI_CLASSREQUEST_H
#define FALCON_WOPI_CLASSREQUEST_H

#include <falcon/module.h>
#include <falcon/class.h>
#include <falcon/pstep.h>

namespace Falcon{
namespace WOPI {

#define FALCON_WOPI_REQUEST_GETS_PROP           "gets"
#define FALCON_WOPI_REQUEST_POSTS_PROP          "posts"
#define FALCON_WOPI_REQUEST_COOKIES_PROP        "cookies"
#define FALCON_WOPI_REQUEST_HEADERS_PROP        "headers"
#define FALCON_WOPI_REQUEST_PARSED_URI_PROP     "parsed_uri"
#define FALCON_WOPI_REQUEST_PROTOCOL_PROP       "protocol"
#define FALCON_WOPI_REQUEST_REQUEST_TIME_PROP   "request_time"
#define FALCON_WOPI_REQUEST_BYTES_SENT_PROP     "bytes_sent"
#define FALCON_WOPI_REQUEST_CONTENT_LENGHT_PROP "content_length"
#define FALCON_WOPI_REQUEST_METHOD_PROP         "method"
#define FALCON_WOPI_REQUEST_CONTENT_TYPE_PROP   "content_type"
#define FALCON_WOPI_REQUEST_CONTENT_ENCODING_PROP   "content_encoding"
#define FALCON_WOPI_REQUEST_AP_AUTH_TYPE_PROP   "ap_auth_type"
#define FALCON_WOPI_REQUEST_USER_PROP           "user"
#define FALCON_WOPI_LOCATION_PROP               "location"
#define FALCON_WOPI_URI_PROP                    "uri"
#define FALCON_WOPI_FILENAME_PROP               "filename"
#define FALCON_WOPI_PATH_INFO_PROP              "path_info"
#define FALCON_WOPI_ARGS_PROP                   "args"
#define FALCON_WOPI_REMOTE_IP_PROP              "remote_ip"
#define FALCON_WOPI_SID_PROP                    "sid"
#define FALCON_WOPI_STARTED_AT_PROP             "startedAt"

class ClassRequest: public Class
{
public:
   ClassRequest();
   virtual ~ClassRequest();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

};

}
}

#endif

/* end of classrequest.h */
