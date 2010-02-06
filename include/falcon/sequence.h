/*
   FALCON - The Falcon Programming Language.
   FILE: sequence.h

   Definition of abstract sequence class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-12-01

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Definition of abstract sequence class.
*/

#ifndef flc_sequence_H
#define flc_sequence_H

#include <falcon/setup.h>
#include <falcon/falcondata.h>

namespace Falcon {

class VMachine;
class Iterator;
class Item;
class Garbageable;

/** Abstract sequence class.
   A sequence is a special user data which is used as internal mean
   by sequence oriented falcon classes. It may be also used by
   extension code to create special lists or generator objects.

   The sequence must be able to create an Iterator object for
   itself. The Iterator will be used internally by the VM or
   eventually wrapped in a Falcon iterator object and then
   given to the script.
*/

class FALCON_DYN_CLASS Sequence: public FalconData
{
   Garbageable* m_owner;
protected:
   mutable Iterator* m_iterList;

public:
   Sequence():
      m_owner(0),
      m_iterList(0)
      {}
      
   virtual ~Sequence()
   {
      invalidateAllIters();
   }
   
   /** Invalidate all the iterators. 
      This disengage all the iterators from this sequence.
      Used when all the sequence becomes invalid.
      
      Iterators are not destroyed (they belong to their owners),
      but they become invalid and stop pointing to this sequence.
   */
   void invalidateAllIters();
   
   /** Invalidate all the iterators but one.
      This method invalidate all the iterators except the one
      provided. This is useful in sequence where destructive
      operations on one iterator (i.e. erase) cause all the
      other unsupported iterators to become invalid.
   */
   void invalidateAnyOtherIter( Iterator* iter );
   
   /** Disposes all the iterators matching a given criterion.
      Useful to dispose all the iterators in a sequence that match
      a certain criterion, as i.e. pointing to an element being deleted.
      
      The criterion is provided by subclasses via the onCriterion() callback.
    */
   virtual void invalidateIteratorOnCriterion() const;
   
   /** Criterion called back by disposeIteratorOnCriterion().
      Return true to remove this iterator, false to let it alive.
      Disposed iterators are removed from the iterator list of this sequence and invalidated.
   */
   virtual bool onCriterion( Iterator* elem ) const {return false;}
   

   /** Mark this class as a sequence. */
   virtual bool isSequence() const { return true; }

   /** Mark this class as a dictionary based sequence. */
   virtual bool isDictionary() const { return false; }

   /** Returns the first element of the sequence.
      If the sequence has not an underlying storage, it may
      generate a temporary item, as the item is immediately
      copied into some destination by the caller.

      Guarantees are taken so that this method is never called
      when v_empty() returns false.
      \return a valid reference to the first item of the sequence
   */
   virtual const Item &front() const = 0;

   /** Returns the first element of the sequence.
      If the sequence has not an underlying storage, it may
      generate a temporary item, as the item is immediately
      copied into some destination by the caller.

      This method is never used by the engine to modify the
      underlying item.

      Guarantees are taken so that this method is never called
      when v_empty() returns false.
      \return a valid reference to the first item of the sequence
   */
   virtual const Item &back() const = 0;

   /** Removes all the items in the sequence. */
   virtual void clear() = 0;

   /** Tells if the series is empty.
      \return false if there is at least one valid item in the series,
         false otherwise.
   */
   virtual bool empty() const =0;

   /** Append an item at the end of the sequence. */
   virtual void append( const Item &data ) = 0;

   /** Prepend an item at the beginning of the sequence. */
   virtual void prepend( const Item &data ) = 0;

   /** Appends to a sequence comprehension.

      Just create the target sequence, and then use this function to
      Fulfill compounds.
      \note DON'T PASS directly parameter pointers, as the stack may be destroyed in the
            meanwhile. Instead, pass copies of the items.
      \param vm The virtual machine where to perform atomic calls.
      \param compounder a range, a sequence or a generator function providing a sequence of data.
      \param filter an optional filter function returning true to accept elemnts, false to discard them
         pass nil if none.

      \TODO: Remove this in the next major release.
   */
   virtual void comprehension( VMachine* vm, const Item& compounder, const Item& filter );

   /**
      Start a comprehension loop.
      
      Creates a stack frame that can be used to iterate ove a multiple comprehension.
      After the return, push this in the stack (as local variables):
      - <filter function> (or nil)
      - <comprehension source 1>
      - ...
      - <comprehension source n>

      And then immediately return.

      Local variables needed for accounting will be already pushed by the comprehension
      startup function, so the stack will probably be partially popualted before
      this method returns.

      The return frame function will take care to generate the comprehension. 
      
      If more than one comprehension source is provided, then each element to be
      stored in this sequence will be a sequence of this kind, with their components
      taken one at a time from each source. For example:
      - Source 1: [a, b, c]
      - Source 2: List( e, f, g )
      - This: Set()
      - Generated elements: Set(a, e), Set(b, f), Set(c, g)
      
      The comprehension terminates when the first source is empty.

      If a filter function is provided, then it's called after each element is created
      and before it is added to this sequence.

      \TODO: Make this virtual in the next major release.
      \return
   */

   /* virtual */ bool comprehension_start( VMachine* vm, const Item& self, const Item& filter );

   /** The sequence may be bound to an object.
    * If the sequence is bound with a falcon script level object,
    * when it receives a gcMark() request, for example, from an iterator
    * referencing it, the related garbageable must be flagged.
    *
    */
   void owner( Garbageable* owner ) { m_owner = owner; }
   Garbageable* owner() { return m_owner; }
   virtual void gcMark( uint32 gen );

   //==============================================================
   // Iterator management
   //

protected:
   friend class Iterator;

   /** Gets an Iterator valid for this sequence.

      If you need an iterator as a pointer or in the target stack,
      use Iterator( Sequence*, bool) instead.

      The iterator constructor calls back this method to be configured.

      It is possible to call this method thereafter to reset the iterator,
      even if it's gone invalid.

      However, it is not legal to call this method with an iterator coming
      from another sequence; this will cause the program to throw a
      CodeError.
      
      \note The base version of this function just adds the iterator to the
      iterator list; it \b MUST be called by all the implementations.

      \param An Iterator to be set.
      \param tail if false, get an iterator to the first element,
         else get an iterator to the last element.
   */
   virtual void getIterator( Iterator& tgt, bool tail = false ) const;

   /** Copy an iterator so that the two points to the same item. 
      
      The source iterator may point to the past-end element, but must not be
      invalid.
      
      \note The base version of this function just adds the iterator to the
         iterator list; it \b MUST be called by all the implementations.
   */
   virtual void copyIterator( Iterator& tgt, const Iterator& source ) const;

   /** Called back to destroy deep data that may be associated with an iterator.
    
      This method is called back at iterator destructor to clear deep data
      that the sequence may have stored in the iterator.
      
      After this call, the iterator is invalidated (if correctly found in the list).
      
      \note The base class version disengage the iterator from the iterator list
            and invalidates it. It normally shouldn't be overloaded by subclasses,
            as the final memory cleaning from a deep iterator, if needed, must be
            separately provided via the Iterator::deletor() interface.
    */
   virtual void disposeIterator( Iterator& tgt ) const;
   

   /** Inserts an element in a position indicated by the iterator.
      The implementation must check that the iterator is a valid
      iterator created by this object and pointing to a valid position.

      Insertion happens at given position, shifting all the remaining
      elements forward; after a successful insert, the iterator must
      point to the newly inserted element, and the previously current
      element is found safely in the next() position of the iterator.

      Valid iterators (generated by this owner) pointing to invalid
      positions must be treated as pointing to last-past-one element;
      insertion causes append on tail, and at return they must be
      valid and point to the last valid element (the one just inserted).

      If the iterator cannot be used, for example because their owner is
      not this item, this method will raise a CodeError.

      \param iterator an iterator.
      \param data the item to be inserted
      \return true if the iterator was valid for this object.
   */
   virtual void insert( Iterator &iter, const Item &data ) = 0;

   /** Deletes the element at position indicated by the iterator.
      The implementation must check that the iterator is a valid
      iterator created by this object and pointing to a valid position.

      Deletion happens at given position, shifting all the remaining
      elements backward; after a successful erase, the iterator must
      point to the element that was previously next in the series,
      or must be invalidated if the removed element was the last.

      If the sequence is empty or the iterator is invalid,
      an AccessError must be thrown. If
      the iterator is referencing another sequence, a CodeError must be
      thrown.

      \param iter an iterator (possibly invalid or not generated by
         this class).
      \return true if the iterator was valid for this object.
   */
   virtual void erase( Iterator &iter ) = 0;
   virtual bool hasNext( const Iterator &iter ) const = 0;
   virtual bool hasPrev( const Iterator &iter ) const = 0;
   virtual bool hasCurrent( const Iterator &iter ) const = 0;

   virtual bool next( Iterator &iter ) const = 0;
   virtual bool prev( Iterator &iter ) const = 0;

   virtual Item& getCurrent( const Iterator &iter ) = 0;
   virtual Item& getCurrentKey( const Iterator &iter ) = 0;

   virtual bool equalIterator( const Iterator &first, const Iterator &second ) const = 0;

};

}

#endif

/* end of sequence.h */
