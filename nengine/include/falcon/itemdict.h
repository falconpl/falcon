/*
   FALCON - The Falcon Programming Language.
   FILE: itemdict.h

   Class storing lexicographic ordered item dictionaries.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 18 Jul 2011 02:22:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef _FALCON_ITEMDICT_H
#define	_FALCON_ITEMDICT_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/overridableclass.h>

namespace Falcon
{

/** Class storing lexicographic ordered item dictionaries.

 */
class FALCON_DYN_CLASS ItemDict
{
public:

   ItemDict();
   ItemDict( const ItemDict& other );
   ~ItemDict();

   ItemDict* clone() const { return new ItemDict(*this); }


   void gcMark( uint32 mark );
   uint32 currentMark() const { return m_currentMark; }

   uint32 flags() const { return m_flags; }
   void flags( uint32 v ) { m_flags = v; }

   void insert( const Item& key, const Item& value );
   void remove( const Item& key );
   Item* find( const Item& key );

   length_t size() const;

   void describe( String& target, int depth = 3, int maxlen = 60 ) const;

   class Enumerator {
   public:
      virtual void operator()( const Item& key, Item& value )=0;
   };
   
   void enumerate( Enumerator& rator );
private:
   class Private;
   Private* _p;
   
   uint32 m_flags;
   uint32 m_currentMark;
};

}

#endif /* _FALCON_ITEMDICT_H_ */

/* end of itemdict.h */
