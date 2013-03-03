/*
   FALCON - The Falcon Programming Language.
   FILE: eventmarshal.h

   A function object specialized in dispatching events from queues
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 21 Feb 2013 08:42:05 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EVENTMARSHAL_H_
#define FALCON_EVENTMARSHAL_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/function.h>
#include <falcon/rootsyntree.h>

#include <falcon/pstep.h>

namespace Falcon
{

/**
Function object specialized in dispatching events from queues.

*/

class FALCON_DYN_CLASS EventMarshal: public Function
{
public:
   EventMarshal( const Item& marshaled );
   EventMarshal( Class* marshalCls, void *marshalInstance );
   virtual ~EventMarshal();

   virtual void invoke( VMContext* ctx, int32 pCount = 0 );
   virtual void gcMark( uint32 mark );
   
private:
   Class* m_cls;
   void* m_instance;

   FALCON_DECLARE_INTERNAL_PSTEP_OWNED(PropResolved, EventMarshal);

};

}

#endif

/* end of eventmarshal.h */
