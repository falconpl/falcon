/*
   FALCON - The Falcon Programming Language.
   FILE: inspector.h

   Helper class for dumping item contents.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Helper class for dumping item contents.

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_INSPECTOR_H
#define FALCON_INSPECTOR_H

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

class TextWriter;
class Item;

/** Helper class for dumping item contents.
 */
class FALCON_DYN_CLASS Inspector
{
public:
   Inspector( TextWriter* tw );
   ~Inspector();

   void inspect( const Item& itm, int32 maxdepth=3, int32 maxsize=128 );

   void inspect_r( const Item& itm, int32 depth, int32 maxdepth, int32 maxsize );

private:
   TextWriter* m_tw;
};

}

#endif

/* end of inspector.h */
