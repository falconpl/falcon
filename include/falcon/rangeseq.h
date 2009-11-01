/*
   FALCON - The Falcon Programming Language.
   FILE: rangeseq.h

   Virtual sequence that can be used to iterate over ranges.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Aug 2009 22:10:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_RANGESEQ_H
#define FALCON_RANGESEQ_H

#include <falcon/setup.h>
#include <falcon/sequence.h>
#include <falcon/item.h>

namespace Falcon {

class FALCON_DYN_CLASS RangeSeq: public Sequence
{
   int64 m_start;
   int64 m_end;
   int64 m_step;

   mutable Item m_number;

public:
   RangeSeq( const CoreRange &rng );
   RangeSeq( int64 s, int64 e, int64 step );
   RangeSeq( const RangeSeq& other ):
      m_start( other.m_start ),
      m_end( other.m_end ),
      m_step( other.m_step )
      {}

   virtual ~RangeSeq();

   virtual const Item &front() const;
   virtual const Item &back() const;
   virtual void clear();
   virtual bool empty() const;
   virtual void append( const Item &data );
   virtual void prepend( const Item &data );
   virtual RangeSeq* clone() const;

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

#endif /* FALCON_RANGESEQ_H */

/* end of rangeseq.h */
