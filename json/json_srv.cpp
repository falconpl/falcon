/*
   FALCON - The Falcon Programming Language.
   FILE: json_srv.cpp

   JSON Service module -- an interface to access the JSON module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 27 Sep 2009 20:58:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/srv/json_srv.h>
#include <falcon/string.h>
#include <falcon/stream.h>
#include <falcon/stringstream.h>
#include <falcon/rosstream.h>

#include "json_mod.h"

namespace Falcon
{

JSONService::JSONService():
   Service( "JSONService" )
{}

bool JSONService::encode( const Item& itm, String& tgt, bool bEncUni, bool bPretty, bool bReadale )
{
   JSON js( bEncUni, bPretty, bReadale );
   StringStream ss;

   if( ! js.encode( itm, &ss ) )
      return false;

   ss.closeToString(tgt);
   return true;

}

bool JSONService::encode( const Item& itm, Stream* tgt, bool bEncUni, bool bPretty, bool bReadale )
{
   JSON js( bEncUni, bPretty, bReadale );
   return js.encode( itm, tgt );
}


bool JSONService::decode( const String& str, Item& tgt )
{
   JSON js;
   ROStringStream ss(str);

   return js.decode( tgt, &ss );
}


bool JSONService::decode( Stream* source, Item& tgt )
{
   JSON js;
   return js.decode( tgt, source );
}

}


/* end of json_srv.cpp */
