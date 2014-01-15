/*
   FALCON - The Falcon Programming Language.
   FILE: client.h

   Web Oriented Programming Interface.

   Generic base class to produce error reports on web.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 15 Jan 2014 12:36:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_WOPI_ERRORHANDLER_H_
#define _FALCON_WOPI_ERRORHANDLER_H_

namespace Falcon {
class String;

namespace WOPI {

class Client;

/** Generic base class to produce error reports on web.
*
*/
class ErrorHandler
{
public:
   ErrorHandler() {}
   virtual ~ErrorHandler() {}
   virtual void replyError( Client* client, int code, const String& message ) = 0;
};

}
}

#endif /* _FALCON_WOPI_ERRORHANDLER_H_ */

/* client.h */
