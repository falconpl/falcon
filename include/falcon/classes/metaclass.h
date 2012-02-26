/*
   FALCON - The Falcon Programming Language.
   FILE: metaclass.h

   Handler for class instances.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 16:04:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_METACLASS_H_
#define _FALCON_METACLASS_H_

#include <falcon/setup.h>
#include <falcon/classes/classmantra.h>

namespace Falcon
{

/** Handler for classes defined by a Falcon script.

 This class implements a class handler for classes a Falcon script. In other words,
 it is a handler for the "class type". The content of this type is a FalconClass,
 where properties and methods declared in a Falcon script class declaration
 are stored.
 
 */
class FALCON_DYN_CLASS MetaClass: public ClassMantra
{
public:

   MetaClass();
   virtual ~MetaClass();
   
   virtual Class* getParent( const String& name ) const;
   virtual bool isDerivedFrom( const Class* cls ) const;
   virtual void enumerateParents( ClassEnumerator& cb ) const;
   virtual void* getParentData( Class* parent, void* data ) const;
   
   void describe( void* instance, String& target, int, int ) const;
   
   //=============================================================
   virtual void op_toString( VMContext* ctx, void* self ) const;
   virtual void op_call( VMContext* ctx, int32 pcount, void* self ) const;
};

}

#endif /* _FALCON_METACLASS_H_ */

/* end of metaclass.h */
