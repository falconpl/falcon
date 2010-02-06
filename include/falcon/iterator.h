/*
   FALCON - The Falcon Programming Language.
   FILE: iterator.h

   Base abstract class for generic collection iterators.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 01 Aug 2009 23:24:09 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Base abstract class for generic collection iterators.
*/

#ifndef flc_iterator_H
#define flc_iterator_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/falcondata.h>
#include <falcon/sequence.h>
#include <falcon/fassert.h>

namespace Falcon {
class Sequence;
class CoreTable;
class Garbageable;
class Item;

/**
   Base abstract class for generic Item sequence/collection iterators.
   
   Iterators and sequences cooperate so that:
   - The iterator provides a common non-abstract non-virtual interface
     to access sequentially sequences of different nature.
   - The underlying sequence provides the virtual functionalities needed
     to access itself and receive orders from its own iterators.
     
   A common usage pattern may be:
   \code
   // ItemArray() supports the sequence interface.
   Sequence* seq = &some_item.asArray()->items();
   ...
   Iterator iter( seq ); // starts at begin.
   while( iter.hasCurrent() )   // while we have some element
   {
      Item& current = iter.getCurrent();
      // ... do something
      
      iter.next()
   }
   \endcode
   
   Iterators support the FalconData interface, so they can be set directly
   as inner data for FalconObject (i.e. Falcon script level iterator instances),
   or be guarded by the garbage collector via the GCPointer garbage pointer shells.
   
   In that case, instead of having iterators in the stack, it's possible to have
   them in the heap (allocated with "new"). 
   
   Iterators can be deleted via a simple delete.
   
   Iterators forward their gcMark() call to the sequence they are linked to, so
   if the sequence is part of a garbage sensible data (CoreArray, CoreObject, GCPointer etc.), 
   the item stays alive untill all the iterators are gone. 
   
   If a sequence is autonomously destroyed (i.e. via an explicit delete on a non-garbage
   sensible data), all the iterators on that sequence are "invalidated".
   
   Iterators are invalidated also when the sequence is changed so that the values
   of some or all iterators become no more valid (i.e. iterators to array or to paged
   maps).
   
   Invalidated iterators do not point to any sequence anymore, and calling any of their
   member except for disposal or isValid() causes a crash.
   
   Iterators are usually able to refer to the target sequence without the need for
   local structures. When this is not possible, their data member is filled with some
   structure specific iterator data, and their deletor() function is set to a function
   that can collect this deep reference data when the iterator is destroyed. As this
   function is totally independent from the source sequence, even deep iterators can
   be destroyed after their sequence has been destroyed, and they have been invalidated.
  
   If the iterator is destroyed while still valid (i.e. because of stack unroll), the
   Sequence::disposeIterator() callback method of the owner sequence is called; this
   gives the sequence the chance to de-account the iterator (and it's usually done in
   the Sequence base class). The disposeIterator() callback must NEVER destroy deep
   data in the iterator, which is done by the deletor() callback as a separate step.
   
   \note The Sequence-iterator pattern works well when there are a limited number of
         active iterator per sequence (1 to 3), which is a common practical usage in 
         many cases. \b Avoid using an iterator-per-item strategy (i.e. back-referenced
         sequence items, able to delete themselves via an erase on their iterator). 
*/
class FALCON_DYN_CLASS Iterator: public FalconData
{

public:
   typedef void(*t_deletor_func)(Iterator*);

private:
   Sequence* m_owner;

   typedef union t_position_holder {
      int64 pos;
      void* ptr;
   } u_idata;

   u_idata m_idata;

   t_deletor_func m_deletor;
   Iterator* m_next;

public:
   Iterator( Sequence* owner, bool tail = false ):
      m_owner( owner ),
      m_deletor(0),
      m_next(0)
   {
      m_idata.ptr = 0;
      owner->getIterator( *this, tail );
   }

   Iterator( const Iterator& other ):
      m_owner( other.m_owner ),
      m_deletor( other.m_deletor ),
      m_next(0)
   {
      m_idata.ptr = 0;
      m_owner->copyIterator( *this, other );
   }


public:
   virtual ~Iterator();
   virtual void gcMark( uint32 mark );
   virtual Iterator *clone() const;

   /** Sets a deep deletor for this iterator.
      Should be called by the Sequence::getIterator and Sequence::copyIterator
      re-implementations in case this iterator is given a deep data that
      must be deleted at termination.
      
      If non-zero, this callback is called at iterator destruction.
   */
   void deletor( t_deletor_func f ) { m_deletor = f; }
   t_deletor_func deletor() const { return m_deletor; }
   /** Advance to the next item.
    * \return false if the iterator couldn't be advanced (because invalid or at end).
    */
   inline bool next() {
      fassert( m_owner != 0 );
      return m_owner->next( *this );
   }

   /** Retreat to the previous item.
    * \return false if the iterator couldn't be retreated (because invalid or at begin).
    */
   inline bool prev() {
      fassert( m_owner != 0 );
      return m_owner->prev( *this );
   }

   inline bool hasNext() const {
      return m_owner != 0 && m_owner->hasNext( *this );
   }

   inline bool hasPrev() const {
      return m_owner != 0 && m_owner->hasPrev( *this );
   }

   /** True if this iterator has a current element.
    *
    * Iterators past the last element (at end) are still valid, but they can't be used
    * to access a current element.
    */
   inline bool hasCurrent() const {
      return m_owner != 0 && m_owner->hasCurrent( *this );
   }

   /** Must be called after an isValid() check */
   inline Item &getCurrent() const {
      fassert( m_owner != 0 );
      return m_owner->getCurrent( *this );
   }

   /** Must be called after an isValid() check.
    * Calling this on non-dictionary sequences will cause an AccessError to be thrown.
    *  */
   inline Item &getCurrentKey() const {
      fassert( m_owner != 0 );
      return m_owner->getCurrentKey( *this );
   }

   inline bool isValid() const { return m_owner != 0; }
   inline bool isOwner( void *collection ) const { return collection == static_cast<void*>(m_owner); }

   inline bool equal( const Iterator &other ) const {
      fassert( m_owner != 0 );
      return m_owner == other.m_owner && m_owner->equalIterator( *this, other );
   }

   /** Erase an element pointed by this iterator.

      This erases the element in the sequence pointed by this iterator.
      the iterator is moved to the next element, or if this was the last
      element in the collection, it points to past-end.

      The operation may or may not invalidate the other iterators on the sequence,
      depending on the nature of the sequecne; however, the iterator itself it's granted
      to stay valid.
   */
   inline void erase() {
      fassert( m_owner != 0 );
      m_owner->erase( *this );
   }

   /** Insert an item at current position in a non-dictionary sequence.
    Calling this on a dictionary sequence will cause an AccessError to be thrown.

    The item is insert before the position to which the iterator points.
    If the iterator isValid() but not hasCurrent(), insertion is at past end (append).

    After the insert, the iterator is not moved (still points to the same element).
    **/
   inline void insert( const Item &item ) {
      fassert( m_owner != 0 );
      m_owner->insert( *this, item );
   }

   /** Turns this in an invalid iterator.
    * Notice that invalid iterators doesn't back-mark their owner.
    */
   void invalidate() { m_owner = 0; }

   Sequence* sequence() const { return m_owner; }

   void sequence( Sequence* s ) { m_owner = s; }

   void *data() const { return m_idata.ptr; }
   void data( void* dt ) { m_idata.ptr = dt; }

   int64 position() const { return m_idata.pos; }
   void position( int64 pos ) { m_idata.pos = pos; }
   
   Iterator* nextIter() const { return m_next; }
   void nextIter( Iterator* n ) { m_next = n; }

   inline void goTop() {
      m_owner->getIterator( *this );
   }

   inline void goBottom() {
      m_owner->getIterator( *this, true );
   }
};

}

#endif /* ITERATOR_H_ */
