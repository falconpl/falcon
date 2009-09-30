/*
   FALCON - The Falcon Programming Language.
   FILE: genericmap.h

   Generic map - a map holding generic values.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago 23 21:55:38 CEST 2004


   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef flc_genericmap_h
#define flc_genericmap_h

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/traits.h>
#include <falcon/basealloc.h>

namespace Falcon
{


typedef struct tag_MapPage {
	uint16 m_count;
	uint16 m_allocated;  // not used by now
	uint16 m_parentElement; // position of the parent element in the parent page
	uint16 m_dummy; // filler
	tag_MapPage *m_parent;
	tag_MapPage *m_higher;
} MAP_PAGE;

class MapIterator;


/** Generic Map.
	This implements a generic map, whose keys and values are of an undetermined
	type and size.

	The map is implemented as a B-tree.

   \note The structure forces 4-bytes alignment, so keys and values may be classes
	instances provided the target architecture supports 4-bytes aligned items.
*/
class FALCON_DYN_CLASS Map: public BaseAlloc
{
   ElementTraits *m_keyTraits;
   ElementTraits *m_valueTraits;

   uint16 m_treeOrder;
   uint16 m_keySize;
   uint16 m_valueSize;

   uint32 m_size;
   MAP_PAGE *m_treeTop;

   friend class MapIterator;

   MAP_PAGE *allocPage() const;

   MAP_PAGE **ptrsOfPage( const MAP_PAGE *ptr ) const;
   void *keysOfPage( const MAP_PAGE *ptr ) const;
   void *valuesOfPage( const MAP_PAGE *ptr ) const;

   MAP_PAGE *ptrInPage( const MAP_PAGE *ptr, uint16 count ) const;
   void *keyInPage( const MAP_PAGE *ptr, uint16 count ) const;
   void *valueInPage( const MAP_PAGE *ptr, uint16 count ) const ;

   bool subFind( const void *key, MapIterator &iter, MAP_PAGE *page ) const;
   bool scanPage( const void *key, MAP_PAGE *currentPage, uint16 count, uint16 &pos ) const;

   void insertSpaceInPage( MAP_PAGE *page, uint16 pos );
   void removeSpaceFromPage( MAP_PAGE *page, uint16 pos );

   void splitPage( MAP_PAGE *page );
   void rebalanceNode( MAP_PAGE *page, MapIterator *scanner = 0 );
	void reshapeChildPointers( MAP_PAGE *page, uint16 startFrom = 0 );

	MAP_PAGE *getRightSibling( const MAP_PAGE *page ) const;
	MAP_PAGE *getLeftSibling( const MAP_PAGE *page ) const;

public:
   Map( ElementTraits *keyt, ElementTraits *valuet, uint16 order = 33 );

   ~Map();
   void destroyPage( MAP_PAGE *page );

   bool insert( const void *key, const void *value);
   bool erase( const void *key );
   MapIterator erase( const MapIterator &iter );
   void *find(const  void *key ) const;

   /** Finds a value or the nearest value possible.
            If the value is found, the function returns true;
            If it's not found, the function returns false and the iterator
            points to the smallest item greater than the given key (so that
            an insert would place the key in the correct
            position).
   */
   bool find( const void *key, MapIterator &iter )const ;

   MapIterator begin() const;
   MapIterator end() const;

   bool empty() const { return m_size == 0; }
   void clear();
   uint32 size() const { return m_size; }
   uint16 order() const { return m_treeOrder; }
};



class FALCON_DYN_CLASS MapIterator: public BaseAlloc
{
   const Map *m_map;
   MAP_PAGE *m_page;
   uint16 m_pagePosition;

   friend class Map;

public:

   MapIterator()
   {}

   MapIterator( const Map *m, MAP_PAGE *p, uint16 ppos):
      m_map(m),
      m_page( p ),
      m_pagePosition( ppos )
   {}

   MapIterator( const MapIterator &other )
   {
      m_map = other.m_map;
      m_page = other.m_page;
      m_pagePosition = other.m_pagePosition;
   }

   bool next();
   bool prev();

   bool hasCurrent() const {
      return m_page != 0 && m_page->m_count > m_pagePosition;
   }

   bool hasNext() const;

   bool hasPrev() const;

   void *currentKey() const
   {
      return m_map->keyInPage( m_page, m_pagePosition );
   }

   void *currentValue() const
   {
      return m_map->valueInPage( m_page, m_pagePosition );
   }

   void currentValue( void* source )
   {
      m_map->m_valueTraits->copy(
            m_map->valueInPage( m_page, m_pagePosition ), source );
   }

   bool equal( const MapIterator &other ) const;
};

class MapPtrTraits: public ElementTraits
{
public:
   virtual ~MapPtrTraits() {}
	virtual uint32 memSize() const;
	virtual void init( void *itemZone ) const;
	virtual void copy( void *targetZone, const void *sourceZone ) const;
	virtual int compare( const void *first, const void *second ) const;
	virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

class MapPtrOwnTraits: public MapPtrTraits
{
public:
   virtual ~MapPtrOwnTraits() {}
	virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

namespace traits
{
	extern FALCON_DYN_SYM MapPtrTraits &t_MapPtr();
	extern FALCON_DYN_SYM MapPtrOwnTraits &t_MapPtrOwn();
}

}

#endif

/* end of genericmap.h */
