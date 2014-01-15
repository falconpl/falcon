/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_filehandler.h

   Micro HTTPD server providing Falcon scripts on the web.

   Handler(s) for requests.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Mar 2010 12:18:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALHTTPD_FILEHANDLER_H_
#define FALHTTPD_FILEHANDLER_H_

#include "falhttpd_rh.h"

namespace Falcon {

class FileHandler: public FalhttpdRequestHandler
{
public:
   FileHandler( const Falcon::String& sFile, FalhttpdClient* client );
   virtual ~FileHandler();
   virtual void serve();
};

}

#endif

/* falhttpd_filehandler.h */
