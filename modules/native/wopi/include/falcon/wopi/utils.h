/*
   FALCON - The Falcon Programming Language.
   FILE: utils.h

   Web Oriented Programming Interface (WOPI)

   Utilities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Feb 2010 14:10:56 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_WOPI_UTILS_H_
#define FALCON_WOPI_UTILS_H_

#include <map>
#include <falcon/string.h>
#include <falcon/uri.h>
#include <falcon/itemdict.h>

namespace Falcon {

class CoreDict;
class CoreObject;

namespace WOPI {
namespace Utils {


typedef std::map<String, String> StringMap;

void fieldsToUriQuery( const ItemDict& fields, String& target );
CoreObject* makeURI( const URI& uri );

void parseQuery( const String &query, ItemDict& dict );
void parseQueryEntry( const String &query, ItemDict& dict );
void addQueryVariable( const String &key, const Item& value, ItemDict& dict );

bool parseHeaderEntry( const String &line, String& key, String& value );
void makeRandomFilename( String& target, int size );
void unescapeQuotes( Falcon::String &str );
void dictAsInputFields( String& fwd, const ItemDict& items );
void htmlEscape( const String& source, String& fwd );
void xrandomize();

}
}
}


#endif /* UTILS_H_ */

/* end of utils.h */

