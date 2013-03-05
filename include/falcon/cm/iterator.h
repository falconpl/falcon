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

class FALCON_DYN_CLASS ClassIterator: public Class
{
public:
   
   ClassIterator();
   virtual ~ClassIterator();

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


   //=============================================================
   //
   virtual void* createInstance() const;   
   virtual bool op_init( VMContext* ctx, void* instance, int pcount ) const;
   
   void op_iter( VMContext* ctx, void* instance ) const;
   void op_next( VMContext* ctx, void* instance ) const;

   FALCON_DECLARE_INTERNAL_PSTEP( IterNext );
   FALCON_DECLARE_INTERNAL_PSTEP( NextNext );
   FALCON_DECLARE_INTERNAL_PSTEP( MethodNext_IterNext );
   FALCON_DECLARE_INTERNAL_PSTEP( MethodNext_NextNext );

   Function* m_Method_next;
};


class FALCON_DYN_CLASS IteratorCarrier
{
public:
   IteratorCarrier( const Item& src ):
      m_source(src),
      m_ready(false),
      m_mark(0)
   {}

   IteratorCarrier():
      m_ready(false),
      m_mark(0)
   {}

   IteratorCarrier( const IteratorCarrier& other ):
      m_source( other.m_source ),
      m_srciter( other.m_srciter ),
      m_ready(other.m_ready),
      m_mark(0)
   {}

   virtual ~IteratorCarrier()
   {
   }

   virtual IteratorCarrier* clone() const { return new IteratorCarrier(*this); }

   const Item& source() const { return m_source; }

private:
   Item m_source;
   Item m_srciter;
   bool m_ready;
   uint32 m_mark;

   friend class Function_rewind;
   friend class Function_next;
   friend class ClassIterator;
};


}
}

#endif

/* end of iterator.h */
