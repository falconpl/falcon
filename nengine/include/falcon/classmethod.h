/*
   FALCON - The Falcon Programming Language.
   FILE: classmethod.h

   Method object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 15:11:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSMETHOD_H_
#define _FALCON_CLASSMETHOD_H_

#include <falcon/setup.h>
#include <falcon/class.h>
#include <falcon/string.h>

#include <falcon/pstep.h>

namespace Falcon
{

/**
 Class handling a method as an item in a falcon script.
 */

class FALCON_DYN_CLASS ClassMethod: public Class
{
public:

   ClassMethod();
   virtual ~ClassMethod();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   virtual void describe( void* instance, String& target, int, int ) const;
   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* self, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* self, const String& prop ) const;
   
   //=============================================================
   virtual void op_create( VMContext* ctx, int32 pcount ) const;
   virtual void op_call( VMContext* ctx, int32 paramCount, void* self ) const;
};

}

#endif /* _FALCON_CLASSMETHOD_H_ */

/* end of classmethod.h */
