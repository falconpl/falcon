/*
   FALCON - The Falcon Programming Language.
   FILE: closure.h

   Closure - function and externally referenced local variabels
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Jan 2012 15:39:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_CLOSURE_H
#define FALCON_CLOSURE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/variable.h>
#include <falcon/closeddata.h>

namespace Falcon {

class VMContext;
class ClassClosure;
class ItemArray;
class Function;

/** Closure abstraction.

 */
class FALCON_DYN_CLASS Closure
{
public:
   Closure();
   Closure( Function* function );
   Closure( const Closure& other );
   virtual ~Closure();

   void gcMark( uint32 mark );
   uint32 gcMark() const { return m_mark; }
   
   Closure* clone() const { return new Closure(*this); }

   virtual void flatten( VMContext* ctx, ItemArray& subItems ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, uint32 pos = 0 );

   const Class* handler() const;

   /** Analyzes the function and the context and closes the needed values.
    \param ctx the context where the closed data is to be found.
    */
   void close( VMContext* ctx );
   Function* closed() const { return m_closed; }
   ClosedData* data() const { return m_data; }

private:
   uint32 m_mark;
   Function* m_closed;
   ClosedData* m_data;

   friend class ClassClosure;
};

}

#endif	/* CLOSURE_H */

/* end of closure.h */
