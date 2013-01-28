/*
   FALCON - The Falcon Programming Language.
   FILE: attributemap.h

   Structure holding attributes for function, classes and modules.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Jan 2013 19:50:55 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_ATTRIBUTEMAP_H
#define FALCON_ATTRIBUTEMAP_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>

namespace Falcon {

class TreeStep;
class DataReader;
class DataWriter;
class ItemArray;
class Attribute;

/** Map holding attributes.
 *
 */
class FALCON_DYN_CLASS AttributeMap
{
public:
   AttributeMap();
   AttributeMap( const AttributeMap& other );
   ~AttributeMap();

   Attribute* add( const String& name );
   Attribute* find( const String& name ) const;

   uint32 size() const;
   Attribute* get( uint32 id ) const;

   void store( DataWriter* stream ) const;
   void restore( DataReader* stream );

   void flatten( ItemArray& subItems ) const;
   void unflatten( const ItemArray& subItems, uint32& start );

private:
   class Private;
   Private* _p;
};

}

#endif   /* FALCON_ATTRIBUTEMAP_H */

/* end of attributemap.h */
