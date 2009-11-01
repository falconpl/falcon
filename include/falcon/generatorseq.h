/*
   FALCON - The Falcon Programming Language.
   FILE: generatorseq.h

   Virtual sequence that can be used to iterate over generators.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 09 Aug 2009 19:04:17 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_GENERATORSEQ_H
#define FALCON_GENERATORSEQ_H

#include <falcon/setup.h>
#include <falcon/sequence.h>
#include <falcon/item.h>

namespace Falcon {

class FALCON_DYN_CLASS GeneratorSeq: public Sequence
{
   VMachine* m_vm;

   Item m_callable;
   mutable Item m_cache_cur;
   mutable Item m_cache_next;

   mutable bool m_bHasCachedCur;
   mutable bool m_bHasCachedNext;
   mutable bool m_bComplete;

   bool fillCurrentValue() const;
   bool fillNextValue() const;

public:
   GeneratorSeq( VMachine* runEvn, const Item& callable );
   GeneratorSeq( const GeneratorSeq& other );

   virtual ~GeneratorSeq();

   virtual const Item &front() const;
   virtual const Item &back() const;
   virtual void clear();
   virtual bool empty() const;
   virtual void append( const Item &data );
   virtual void prepend( const Item &data );
   virtual GeneratorSeq* clone() const;
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

#endif /* FALCON_GENERATORSEQ_H */

/* end of generatorseq.h */
