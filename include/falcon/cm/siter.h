/*
   FALCON - The Falcon Programming Language.
   FILE: iterator.h

   Falcon core module -- Iterator
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Feb 2013 21:20:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_ITERATOR_H
#define FALCON_CORE_ITERATOR_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/class.h>
#include <falcon/item.h>

namespace Falcon {

class Function;

namespace Ext {

class Function_rewind;
class Function_next;


class FALCON_DYN_CLASS SIterCarrier
{
public:
   SIterCarrier( const Item& src ):
      m_source(src),
      m_hasCurrent(false),
      m_bIsFirst(true),
      m_mark(0)
   {
      m_current.setDoubt();
   }

   SIterCarrier():
      m_hasCurrent(false),
      m_bIsFirst(true),
      m_mark(0)
   {
      m_current.setDoubt();
   }

   SIterCarrier( const SIterCarrier& other ):
      m_source( other.m_source ),
      m_current( other.m_current ),
      m_hasCurrent(other.m_hasCurrent),
      m_bIsFirst(other.m_bIsFirst),
      m_mark(0)
   {}

   virtual ~SIterCarrier()
   {
   }

   virtual SIterCarrier* clone() const { return new SIterCarrier(*this); }

   bool isFirst() const { return m_bIsFirst; }
   void isFirst( bool v ) { m_bIsFirst = v; }
   bool hasCurrent() const { return (! m_bIsFirst) && m_hasCurrent; }
   void hasCurrent(bool b) { m_hasCurrent = b; }
   bool hasNext() const { return (m_bIsFirst || m_current.isDoubt()) && ! m_current.isBreak(); }

   Item& source() { return m_source; }
   Item& intenralIter() { return m_internalIter; }
   Item& current() { return m_current; }

   uint32 currentMark() const { return m_mark; }
   void gcMark( uint32 m ) {
      if ( m != m_mark )
      {
         m_mark = m;
         m_source.gcMark(m);
         m_internalIter.gcMark(m);
         m_current.gcMark(m);
      }
   }

private:
   Item m_source;
   Item m_internalIter;
   Item m_current;
   bool m_hasCurrent;
   bool m_bIsFirst;
   uint32 m_mark;
};



class FALCON_DYN_CLASS ClassSIter: public Class
{
public:

   ClassSIter();
   virtual ~ClassSIter();

   void invokeDirectNextMethod( VMContext* ctx, void* instance, int32 pcount );

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* insatnce ) const;
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   virtual void store( VMContext*, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext*, DataReader* stream ) const;

   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   // Start the iteration on a configured carrier
   // it calls the op_iter of the ic->source field, and puts an appropriate after-step
   // CAUTION: the afterstep returns the frame in ctx (return value is "self").
   void startIteration( SIterCarrier* ic, VMContext* ctx );

   //=============================================================
   //
   virtual void* createInstance() const;

   void op_iter( VMContext* ctx, void* instance ) const;
   void op_next( VMContext* ctx, void* instance ) const;

   FALCON_DECLARE_INTERNAL_PSTEP( IterNext );
   FALCON_DECLARE_INTERNAL_PSTEP( NextNext );
   FALCON_DECLARE_INTERNAL_PSTEP( StartIterNext );
   FALCON_DECLARE_INTERNAL_PSTEP( StartNextNext );
   FALCON_DECLARE_INTERNAL_PSTEP( MethodNext_NextNext );

   Function* m_Method_next;
};


}
}

#endif

/* end of iterator.h */
