/*
   FALCON - The Falcon Programming Language.
   FILE: coredict.h

   Standard language dictionary object handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 14:15:37 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_COREDICT_H_
#define _FALCON_COREDICT_H_

#include <falcon/setup.h>
#include <falcon/class.h>

#include <falcon/pstep.h>

namespace Falcon
{

/**
 Class handling a dictionary as an item in a falcon script.
 */

class FALCON_DYN_CLASS CoreDict: public Class
{
public:

   class cpars {
   public:
      cpars():
         m_other(0),
         m_bCopy(false)
      {}

      cpars( void* other, bool bCopy ):
         m_other(other),
         m_bCopy(false)
      {}

      void* m_other;
      bool m_bCopy;
   };

   CoreDict();
   virtual ~CoreDict();

   //=============================================================

   virtual void* create( void* creationParams ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;
   virtual void describe( void* instance, String& target ) const;

   virtual void gcMark( void* self, uint32 mark ) const;
   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;

   //virtual int compare( void* self, const Item& value ) const;

   //=============================================================

   virtual void op_add( VMachine *vm, void* self ) const;
   virtual void op_isTrue( VMachine *vm, void* self ) const;
   virtual void op_toString( VMachine *vm, void* self ) const;

   virtual void op_getProperty( VMachine *vm, void* self, const String& prop) const;
   virtual void op_getIndex(VMachine *vm, void* self ) const;
   virtual void op_setIndex(VMachine *vm, void* self ) const;

private:

   class FALCON_DYN_CLASS ToStringNextOp: public PStep {
   public:
      ToStringNextOp();
      static void apply_( const PStep*, VMachine* vm );
   } m_toStringNextOp;

};

}

#endif /* _FALCON_COREDICT_H_ */

/* end of coredict.h */
