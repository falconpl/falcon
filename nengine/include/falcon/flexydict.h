/*
   FALCON - The Falcon Programming Language.
   FILE: flexydict.cpp

   Standard item type for flexy dictionaries of property-items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 16:55:07 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FLEXYDICT_H
#define	_FALCON_FLEXYDICT_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/itemarray.h>

namespace Falcon {

/** Standard item type for flexy dictionaries of property-items.

 This is used for both prototype and flexy class items.

 \note The whole class is inlined when defining FALCON_FLEXYDICT_PRIVATE_INTERNAL

 \todo Optimize this class using advanced map structures.
 */
class FlexyDict
{
public:

   FlexyDict();

   FlexyDict( const FlexyDict& other );

   ~FlexyDict();

   void gcMark( uint32 mark );

   void enumerateProps( Class::PropertyEnumerator& e ) const;

   void enumeratePV( Class::PVEnumerator& e );

   /** Verifies if a property is in this dictionary.
    Warning: this doesn't search the properties in the bases.
    Searching properties in the bases is delegated to the VM if necessary.
    */
   bool hasProperty( const String& p ) const;

   void describe( String& target, int depth, int maxlen ) const;

   /** Gets an item stored in this flexy dictionary.
    Warning: this doesn't search the properties in the bases.
    Searching properties in the bases is delegated to the VM if necessary.
    */
   Item* find( const String& value );

   void insert( const String& key, Item& value );
   inline uint32 currentMark() const { return m_currentMark; }

   const ItemArray& base() const { return m_base; }
   ItemArray& base() { return m_base; }

private:
   class Private;
   Private* _p;

   uint32 m_currentMark;
   uint32 m_flags;

   ItemArray m_base;
};

}

#endif	/* _FALCON_FLEXYDICT_H */

/* end of flexydict.h */
