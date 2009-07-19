/*
   FALCON - The Falcon Programming Language.
   FILE: genericvector.h

   Generic vector - a generic vector of elements
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven oct 27 11:02:00 CEST 2006


   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef flc_genericvector_h
#define flc_genericvector_h

#include <falcon/setup.h>
#include <falcon/traits.h>
#include <falcon/basealloc.h>

namespace Falcon
{

/** Generic typeless vector.
   The allocated size of a generic vector is always the needed size + 1.
   This is because if you need to push a pointer from the same vector, you can push it
   and THEN reallocate it.
*/


class FALCON_DYN_CLASS GenericVector: public BaseAlloc
{
	byte *m_data;
	uint32 m_size;
	uint32 m_allocated;
	uint32 m_threshold_size;

   typedef enum {
      alloc_block = 32
   } consts;

protected:
	uint32 m_itemSize;
	const ElementTraits *m_traits;

	GenericVector():
	   m_data(0),
	   m_size(0),
	   m_allocated(0),
	   m_threshold_size(0)
	{}

	void init( const ElementTraits *traits, uint32 prealloc );

public:

	GenericVector( const ElementTraits *traits, uint32 prealloc=0 );
	~GenericVector();

	void insert( void *data, uint32 pos );
	bool remove( uint32 pos );
	void *at( uint32 pos ) const { return m_data + ( pos * m_itemSize ); }
	void set( void *data, uint32 pos );
	void push( void *data );
	void pop() { m_size --; }

	void *top() const { return m_data + ( (m_itemSize) * (m_size-1) ); }
   void reserve( uint32 s );
   void resize( uint32 s );

   void threshHold( uint32 size ) { m_threshold_size = size; }
   uint32 threshHold() const { return m_threshold_size; }

   uint32 size() const { return m_size; }
   bool empty() const { return m_size == 0; }
};

}

#endif

/* end of genericvector.h */
