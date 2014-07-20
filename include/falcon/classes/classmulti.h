/*
   FALCON - The Falcon Programming Language.
   FILE: multiclass.h

   Base class for classes holding more subclasses.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 06:35:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_MULTICLASS_H_
#define _FALCON_MULTICLASS_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/classes/classmantra.h>
#include <falcon/pstep.h>

namespace Falcon
{

/** Base class for classes holding more subclasses.

 Common code used by classes that hold multiple sub-classes.

 \note This is still an abstract base class.

 */
class FALCON_DYN_CLASS ClassMulti: public ClassMantra
{
public:
   
   ClassMulti( const String& name );
   ClassMulti( const String& name, int TypeID );
   virtual ~ClassMulti();

   virtual bool getProperty( const String& name, void* instance, Item& target ) const;
   virtual bool setProperty( const String& name, void* instance, const Item& target ) const;

   //=========================================================
   // Operators.
   //

   virtual void op_neg( VMContext* ctx, void* self ) const;
   virtual void op_add( VMContext* ctx, void* self ) const;
   virtual void op_sub( VMContext* ctx, void* self ) const;
   virtual void op_mul( VMContext* ctx, void* self ) const;
   virtual void op_div( VMContext* ctx, void* self ) const;
   virtual void op_mod( VMContext* ctx, void* self ) const;
   virtual void op_pow( VMContext* ctx, void* self ) const;
   virtual void op_shl( VMContext* ctx, void* self ) const;
   virtual void op_shr( VMContext* ctx, void* self ) const;
   virtual void op_aadd( VMContext* ctx, void* self) const;
   virtual void op_asub( VMContext* ctx, void* self ) const;
   virtual void op_amul( VMContext* ctx, void* self ) const;
   virtual void op_adiv( VMContext* ctx, void* self ) const;
   virtual void op_amod( VMContext* ctx, void* self ) const;
   virtual void op_apow( VMContext* ctx, void* self ) const;
   virtual void op_ashr( VMContext* ctx, void* self ) const;
   virtual void op_ashl( VMContext* ctx, void* self ) const;
   virtual void op_inc( VMContext* ctx, void* self ) const;
   virtual void op_dec( VMContext* ctx, void* self) const;
   virtual void op_incpost( VMContext* ctx, void* self ) const;
   virtual void op_decpost( VMContext* ctx, void* self ) const;
   virtual void op_getIndex( VMContext* ctx, void* self ) const;
   virtual void op_setIndex( VMContext* ctx, void* self ) const;
   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* self, const String& prop ) const;
   virtual void op_compare( VMContext* ctx, void* self ) const;
   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_in( VMContext* ctx, void* self ) const;
   virtual void op_call( VMContext* ctx, int32 paramCount, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;
   virtual void op_iter( VMContext* ctx, void* self ) const;
   virtual void op_next( VMContext* ctx, void* self ) const;

protected:

   class Property
   {
   public:
      Class* m_provider;
      int m_itemId;

      Property()
      {}

      Property( Class* cls, int itemId ):
         m_provider( cls ),
         m_itemId( itemId )
      {}
   };

   Property** m_overrides;

   class Private_base;
   Private_base* _p_base;

   void checkAddOverride( const String& propName, Property* p );
   void checkRemoveOverride( const String& propName );
   bool getOverride( void* self, int op, Class*& cls, void*& udata ) const;

private:
   inline bool inl_get_override( void* self, int op, Class*& cls, void*& udata ) const;

};

}

#endif /* _FALCON_MULTICLASS_H_ */

/* end of multiclass.h */
