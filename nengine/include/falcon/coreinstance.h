/*
   FALCON - The Falcon Programming Language.
   FILE: coreinstance.h

   Hander instances created out of classes defined by scripts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 16:04:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_COREINSTANCE_H_
#define _FALCON_COREINSTANCE_H_

#include <falcon/setup.h>
#include <falcon/class.h>


#define OVERRIDE_OP_NEG       "__neg"

#define OVERRIDE_OP_ADD       "__add"
#define OVERRIDE_OP_SUB       "__sub"
#define OVERRIDE_OP_MUL       "__mul"
#define OVERRIDE_OP_DIV       "__div"
#define OVERRIDE_OP_MOD       "__mod"
#define OVERRIDE_OP_POW       "__pow"

#define OVERRIDE_OP_AADD      "__aadd"
#define OVERRIDE_OP_ASUB      "__asub"
#define OVERRIDE_OP_AMUL      "__amul"
#define OVERRIDE_OP_ADIV      "__adiv"
#define OVERRIDE_OP_AMOD      "__amod"
#define OVERRIDE_OP_APOW      "__apow"

#define OVERRIDE_OP_INC       "__inc"
#define OVERRIDE_OP_DEC       "__dec"
#define OVERRIDE_OP_INCPOST   "__incpost"
#define OVERRIDE_OP_DECPOST   "__decpost"

#define OVERRIDE_OP_CALL      "__call"

#define OVERRIDE_OP_GETINDEX  "__getIndex"
#define OVERRIDE_OP_SETINDEX  "__setIndex"
#define OVERRIDE_OP_GETPROP   "__getProperty"
#define OVERRIDE_OP_SETPROP   "__setProperty"

#define OVERRIDE_OP_COMPARE   "__compare"
#define OVERRIDE_OP_ISTRUE    "__isTrue"
#define OVERRIDE_OP_IN        "__in"
#define OVERRIDE_OP_PROVIDES  "__provides"


namespace Falcon
{

class FalconClass;

/** Hander instances created out of classes defined by scripts.

 This class handles an instance of a Falcon class (otherwise known as
 "Falcon object").

 */
class FALCON_DYN_CLASS CoreInstance: public Class
{
public:
   CoreInstance();
   virtual ~CoreInstance();

   /** Creation params for core instances.
      Require a FalconClass from where to originate the item.
    */
   class cpars {
   public:
      FalconClass* flc;

      cpars( FalconClass* f ):
         flc(f)
         {}
   };
   
   //=========================================
   // Instance management
   
   virtual void* create( void* creationParams=0 ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   //=========================================================
   // Class management
   //

   virtual void gcMark( void* self, uint32 mark ) const;
   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;
   virtual bool hasProperty( void* self, const String& prop ) const;
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   //=========================================================
   // Operators.
   //

   virtual void op_neg( VMachine *vm, void* self ) const;
   virtual void op_add( VMachine *vm, void* self ) const;
   virtual void op_sub( VMachine *vm, void* self ) const;
   virtual void op_mul( VMachine *vm, void* self ) const;
   virtual void op_div( VMachine *vm, void* self ) const;
   virtual void op_mod( VMachine *vm, void* self ) const;
   virtual void op_pow( VMachine *vm, void* self ) const;
   virtual void op_aadd( VMachine *vm, void* self) const;
   virtual void op_asub( VMachine *vm, void* self ) const;
   virtual void op_amul( VMachine *vm, void* self ) const;
   virtual void op_adiv( VMachine *vm, void* self ) const;
   virtual void op_amod( VMachine *vm, void* self ) const;
   virtual void op_apow( VMachine *vm, void* self ) const;
   virtual void op_inc(VMachine *vm, void* self ) const;
   virtual void op_dec(VMachine *vm, void* self) const;
   virtual void op_incpost(VMachine *vm, void* self ) const;
   virtual void op_decpost(VMachine *vm, void* self ) const;
   virtual void op_getIndex(VMachine *vm, void* self ) const;
   virtual void op_setIndex(VMachine *vm, void* self ) const;
   virtual void op_getProperty( VMachine *vm, void* self, const String& prop) const;
   virtual void op_setProperty( VMachine *vm, void* self, const String& prop ) const;
   virtual void op_compare( VMachine *vm, void* self ) const;
   virtual void op_isTrue( VMachine *vm, void* self ) const;
   virtual void op_in( VMachine *vm, void* self ) const;
   virtual void op_provides( VMachine *vm, void* self, const String& property ) const;
   virtual void op_call( VMachine *vm, int32 paramCount, void* self ) const;
   virtual void op_toString( VMachine *vm, void* self ) const;
};

}

#endif /* _FALCON_COREINSTANCE_H_ */

/* end of coreinstance.h */
