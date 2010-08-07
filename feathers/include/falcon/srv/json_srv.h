/*
   FALCON - The Falcon Programming Language.
   FILE: json_srv.h

   JSON Service module -- an interface to access the JSON module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 27 Sep 2009 20:58:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef flc_json_srv_H
#define flc_json_srv_H

#include <falcon/setup.h>
#include <falcon/service.h>
#include <falcon/string.h>
#include <falcon/stream.h>

namespace Falcon
{

#define JSONSERVICE_NAME "JSONService"

/** Publishes the JSON converter service as "JSONService".
 *
 */
class JSONService: public Service
{
public:
   JSONService();

   virtual bool encode( const Item& itm, String& tgt, bool bEncUni = false, bool bPretty=false, bool bReadale = false );
   virtual bool encode( const Item& itm, Stream* tgt, bool bEncUni = false, bool bPretty=false, bool bReadale = false );

   virtual bool decode( const String& str, Item& tgt );
   virtual bool decode( Stream* source, Item& tgt );
};

}

#endif

/* end of json_srv.h */
