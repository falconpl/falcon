/*
   FALCON - The Falcon Programming Language.
   FILE: poopseq.h

   Virtual sequence that can be used to iterate over poop providers.
   *AT THE MOMENT* providing just "append" method to re-use sequence
   comprehension in OOP and POOP contexts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 10 Aug 2009 10:53:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_POOPSEQ_H
#define FALCON_POOPSEQ_H

#include <falcon/setup.h>
#include <falcon/sequence.h>
#include <falcon/item.h>

namespace Falcon {

class FALCON_DYN_CLASS PoopSeq: public Sequence
{
   Item m_appendMth;
   uint32 m_mark;
   VMachine* m_vm;

public:
   PoopSeq( VMachine* vm, const Item &iobj );
   PoopSeq( const PoopSeq& other );
   virtual ~PoopSeq();

   virtual const Item &front() const;
   virtual const Item &back() const;
   virtual void clear();
   virtual bool empty() const;
   virtual void append( const Item &data );
   virtual void prepend( const Item &data );
   virtual PoopSeq* clone() const;
   virtual void gcMark( uint32 gen );

   //==============================================================
   // Iterator management
   //

protected:
   virtual void getIterator( Iterator& tgt, bool tail = false ) const;
   virtual void copyIterator( Iterator& tgt, const Iterator& source ) const;
   virtual void insert( Iterator &iter, const Item &data );
   virtual void erase( Iterator &iter );
   virtual bool hasNext( const Iterator &iter ) const;
   virtual bool hasPrev( const Iterator &iter ) const;
   virtual bool hasCurrent( const Iterator &iter ) const;

   virtual bool next( Iterator &iter ) const;
   virtual bool prev( Iterator &iter ) const;

   virtual Item& getCurrent( const Iterator &iter );
   virtual Item& getCurrentKey( const Iterator &iter );

   virtual bool equalIterator( const Iterator &first, const Iterator &second ) const;
};

}

#endif

/* end of poop.h */
