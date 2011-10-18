/*
   FALCON - The Falcon Programming Language.
   FILE: classgeneric.h

   Generic object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 12 Aug 2011 19:37:30 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSGENERIC_H_
#define _FALCON_CLASSGENERIC_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{

/** Class handling GenericItem subclasses.
 
 The Class-Data pair can easily handle heterogeneous objects and arbitrary
 data coming from outside the Falcon engine. However, not all the data requires
 to be that flexible. 
 
 This class handles a hierarcy of "generic" objects that share the same GenericItem
 base class. When the item doesn't require particular specific management,
 but it just requires to be shared with the engine through common and simple rules,
 and it is possible to inherit the object from the GenericItem base class, then
 using this handler class can be a correct solution.
 
 \note Do not use this class if you need to access object properties, or to
 override some operators. Prefer the ClassUser solution, which is more flexible
 and adequate.
 
 \note Generic items are not serializable, but they can be cloneable.
 */

class FALCON_DYN_CLASS ClassGeneric: public Class
{
public:

   ClassGeneric();
   virtual ~ClassGeneric();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* self ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;
   virtual void describe( void* self, String& target, int depth = 3, int maxlen = 60 ) const;

   virtual void gcMark( void* instance, uint32 mark ) const;
   virtual bool gcCheck( void* instance, uint32 mark ) const;
   
   //=============================================================
};

}

#endif /* _FALCON_CLASSGENERIC_H_ */

/* end of classgeneric.h */
