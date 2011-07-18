/*
   FALCON - The Falcon Programming Language.
   FILE: flexyclass.h

   Class handling flexible objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 18 Jul 2011 02:22:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FLEXYCLASS_H_
#define _FALCON_FLEXYCLASS_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/overridableclass.h>

namespace Falcon
{

/** Class handling flexible objects.

 */
class FALCON_DYN_CLASS FlexyClass: public OverridableClass
{
public:

   FlexyClass();
   virtual ~FlexyClass();

   //====================================================================
   // Overrides from Class
   //
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   //=========================================================
   // Class management
   //

   virtual void gcMark( void* self, uint32 mark ) const;

   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* self, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* self, const String& prop ) const;
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   //=========================================================
   // Operators.
   //

   virtual void op_create( VMContext* ctx, int32 pcount ) const;
   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* self, const String& prop ) const;
};

}

#endif /* _FALCON_FLEXYCLASS_H_ */

/* end of flexyclass.h */
