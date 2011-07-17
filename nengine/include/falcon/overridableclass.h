/*
   FALCON - The Falcon Programming Language.
   FILE: overridableclass.h

   Base abstract class for classes providing an override system.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 16:04:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_OVERRIDABLECLASS_H_
#define _FALCON_OVERRIDABLECLASS_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/class.h>

namespace Falcon
{

class Function;

/** Base abstract class for classes providing an override system.

 The classes derived by this class are willing to offer their instance
 the ability to override default operators.

 This class offers a simple and standardized mechanism to override operators.

 FalconClass and FlexyClass are two instances of this class.
 */
class FALCON_DYN_CLASS OverridableClass: public Class
{
public:

   OverridableClass( const String& name );
   OverridableClass( const String& name, int64 tid );
   virtual ~OverridableClass();

   virtual void op_neg( VMContext* ctx, void* self ) const;
   virtual void op_add( VMContext* ctx, void* self ) const;
   virtual void op_sub( VMContext* ctx, void* self ) const;
   virtual void op_mul( VMContext* ctx, void* self ) const;
   virtual void op_div( VMContext* ctx, void* self ) const;
   virtual void op_mod( VMContext* ctx, void* self ) const;
   virtual void op_pow( VMContext* ctx, void* self ) const;
   virtual void op_aadd( VMContext* ctx, void* self) const;
   virtual void op_asub( VMContext* ctx, void* self ) const;
   virtual void op_amul( VMContext* ctx, void* self ) const;
   virtual void op_adiv( VMContext* ctx, void* self ) const;
   virtual void op_amod( VMContext* ctx, void* self ) const;
   virtual void op_apow( VMContext* ctx, void* self ) const;
   virtual void op_inc( VMContext* ctx, void* self ) const;
   virtual void op_dec( VMContext* ctx, void* self) const;
   virtual void op_incpost( VMContext* ctx, void* self ) const;
   virtual void op_decpost( VMContext* ctx, void* self ) const;
   virtual void op_getIndex( VMContext* ctx, void* self ) const;
   virtual void op_setIndex( VMContext* ctx, void* self ) const;
   // won't provide set/get property

   virtual void op_compare( VMContext* ctx, void* self ) const;
   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_in( VMContext* ctx, void* self ) const;
   virtual void op_provides( VMContext* ctx, void* self, const String& property ) const;
   virtual void op_call( VMContext* ctx, int32 paramCount, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;

protected:
   /** Records the method if the name is that of an override method.
    \param mth A method that is being added to this class.

    If the name of the method corresponds to an override name,
    the override entity will be set.
    */
   void overrideAddMethod( const String& name, Function* mth );

   /** Checks if a method is an override method.
    \param mth A method that is being added to this class.

    If the name of the method corresponds to an override name,
    the override entity will be reset.
    */
   void overrideRemoveMethod( const String& name );

   bool overrideGetProperty( VMContext* ctx, void* self, const String& propName ) const;
   bool overrideSetProperty( VMContext* ctx, void* self, const String& propName ) const;

   inline void override_unary( VMContext* ctx, void*, int op_id, const String& opName ) const;
   inline void override_binary( VMContext* ctx, void*, int op_id, const String& opName ) const;

   Function** m_overrides;
};

}

#endif /* _FALCON_OVERRIDABLECLASS_H_ */

/* end of overridableclass.h */
