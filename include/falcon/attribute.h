/*
   FALCON - The Falcon Programming Language.
   FILE: attribute.h

   Structure holding attributes for function, classes and modules.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Jan 2013 19:50:55 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_ATTRIBUTE_H
#define FALCON_ATTRIBUTE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/item.h>


namespace Falcon {

class TreeStep;
class DataReader;
class DataWriter;
class ItemArray;

/** Attribute class.

    This structure holds the representation of an attribute,
    that is, a pair of name-value (where the value is a Falcon Item)
    associated with mantras or modules, which is initialized at module
    load time through a generator expression.

 */
class FALCON_DYN_CLASS Attribute
{
public:
   static const char* CLASS_NAME;

   Attribute();
   Attribute( uint32 id, const String& name, TreeStep* gen, const Item& dflt );
   Attribute( const Attribute& other );

   ~Attribute();

   const String& name() const { return m_name; }
   void name( const String& n ) { m_name = n; }

   TreeStep* generator() const { return m_generator; }
   void generator( TreeStep* gen );

   const Item& value() const { return m_value; }
   Item& value() { return m_value; }

   void currentMark() const { m_name.currentMark();  }
   void gcMark( uint32 mark );

   void store( DataWriter* stream ) const;
   void restore( DataReader* stream );
   void flatten( ItemArray& subItems ) const;
   void unflatten( const ItemArray& subItems, uint32& start );

   uint32 id() const { return m_id; }
   void id( uint32 i ) { m_id = i; }
private:
   int m_id;
   String m_name;
   TreeStep* m_generator;
   Item m_value;
};

}

#endif	/* FALCON_ATTRIBUTE_H */

/* end of attribute.h */
