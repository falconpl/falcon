/*
   FALCON - The Falcon Programming Language.
   FILE: multiclass.cpp

   Base class for classes holding more subclasses.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 06:35:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/multiclass.cpp"

#include <falcon/classes/classmulti.h>
#include <falcon/itemarray.h>
#include <falcon/function.h>
#include <falcon/ov_names.h>
#include <falcon/vmcontext.h>
#include <falcon/fassert.h>

#include <falcon/stderrors.h>

#include <cstring>

#include "classmulti_private.h"

namespace Falcon {

ClassMulti::ClassMulti( const String& name, int typeId ):
   ClassMantra( name, typeId )
{
   m_overrides = new Property*[OVERRIDE_OP_COUNT];
   memset( m_overrides, 0, OVERRIDE_OP_COUNT* sizeof( Property* ));
}

ClassMulti::ClassMulti( const String& name ):
   ClassMantra( name, FLC_ITEM_USER )
{
   m_overrides = new Property*[OVERRIDE_OP_COUNT];
   memset( m_overrides, 0, OVERRIDE_OP_COUNT* sizeof( Property* ));
}

ClassMulti::~ClassMulti()
{
   delete m_overrides;
}


void ClassMulti::checkAddOverride( const String& name, Property* p )
{
   if( name == OVERRIDE_OP_NEG ) m_overrides[OVERRIDE_OP_NEG_ID] = p;

   else if( name == OVERRIDE_OP_ADD ) m_overrides[OVERRIDE_OP_ADD_ID] = p;
   else if( name == OVERRIDE_OP_SUB ) m_overrides[OVERRIDE_OP_SUB_ID] = p;
   else if( name == OVERRIDE_OP_MUL ) m_overrides[OVERRIDE_OP_MUL_ID] = p;
   else if( name == OVERRIDE_OP_DIV ) m_overrides[OVERRIDE_OP_DIV_ID] = p;
   else if( name == OVERRIDE_OP_MOD ) m_overrides[OVERRIDE_OP_MOD_ID] = p;
   else if( name == OVERRIDE_OP_POW ) m_overrides[OVERRIDE_OP_POW_ID] = p;
   else if( name == OVERRIDE_OP_SHR ) m_overrides[OVERRIDE_OP_SHR_ID] = p;
   else if( name == OVERRIDE_OP_SHL ) m_overrides[OVERRIDE_OP_SHL_ID] = p;

   else if( name == OVERRIDE_OP_AADD ) m_overrides[OVERRIDE_OP_AADD_ID] = p;
   else if( name == OVERRIDE_OP_ASUB ) m_overrides[OVERRIDE_OP_ASUB_ID] = p;
   else if( name == OVERRIDE_OP_AMUL ) m_overrides[OVERRIDE_OP_AMUL_ID] = p;
   else if( name == OVERRIDE_OP_ADIV ) m_overrides[OVERRIDE_OP_ADIV_ID] = p;
   else if( name == OVERRIDE_OP_AMOD ) m_overrides[OVERRIDE_OP_AMOD_ID] = p;
   else if( name == OVERRIDE_OP_APOW ) m_overrides[OVERRIDE_OP_APOW_ID] = p;
   else if( name == OVERRIDE_OP_ASHR ) m_overrides[OVERRIDE_OP_ASHR_ID] = p;
   else if( name == OVERRIDE_OP_ASHL ) m_overrides[OVERRIDE_OP_ASHL_ID] = p;

   else if( name == OVERRIDE_OP_INC ) m_overrides[OVERRIDE_OP_INC_ID] = p;
   else if( name == OVERRIDE_OP_DEC ) m_overrides[OVERRIDE_OP_DEC_ID] = p;
   else if( name == OVERRIDE_OP_INCPOST ) m_overrides[OVERRIDE_OP_INCPOST_ID] = p;
   else if( name == OVERRIDE_OP_DECPOST ) m_overrides[OVERRIDE_OP_DECPOST_ID] = p;

   else if( name == OVERRIDE_OP_CALL ) m_overrides[OVERRIDE_OP_CALL_ID] = p;

   else if( name == OVERRIDE_OP_GETINDEX ) m_overrides[OVERRIDE_OP_GETINDEX_ID] = p;
   else if( name == OVERRIDE_OP_SETINDEX ) m_overrides[OVERRIDE_OP_SETINDEX_ID] = p;
   else if( name == OVERRIDE_OP_GETPROP ) m_overrides[OVERRIDE_OP_GETPROP_ID] = p;
   else if( name == OVERRIDE_OP_SETPROP ) m_overrides[OVERRIDE_OP_SETPROP_ID] = p;

   else if( name == OVERRIDE_OP_COMPARE ) m_overrides[OVERRIDE_OP_COMPARE_ID] = p;
   else if( name == OVERRIDE_OP_ISTRUE ) m_overrides[OVERRIDE_OP_ISTRUE_ID] = p;
   else if( name == OVERRIDE_OP_IN ) m_overrides[OVERRIDE_OP_IN_ID] = p;
   else if( name == OVERRIDE_OP_TOSTRING ) m_overrides[OVERRIDE_OP_TOSTRING_ID] = p;
   else if( name == OVERRIDE_OP_ITER ) m_overrides[OVERRIDE_OP_ITER_ID] = p;
   else if( name == OVERRIDE_OP_NEXT ) m_overrides[OVERRIDE_OP_NEXT_ID] = p;
   else if( name == OVERRIDE_OP_UNKMSG ) m_overrides[OVERRIDE_OP_UNKMSG_ID] = p;

#if OVERRIDE_OP_UNKMSG_ID + 1 != OVERRIDE_OP_COUNT
#error "You forgot to update the operator overrides in MultiClass::checkAddOverride"
#endif

}


void ClassMulti::checkRemoveOverride( const String& name )
{
   if( name == OVERRIDE_OP_NEG ) m_overrides[OVERRIDE_OP_NEG_ID] = 0;

   else if( name == OVERRIDE_OP_ADD ) m_overrides[OVERRIDE_OP_ADD_ID] = 0;
   else if( name == OVERRIDE_OP_SUB ) m_overrides[OVERRIDE_OP_SUB_ID] = 0;
   else if( name == OVERRIDE_OP_MUL ) m_overrides[OVERRIDE_OP_MUL_ID] = 0;
   else if( name == OVERRIDE_OP_DIV ) m_overrides[OVERRIDE_OP_DIV_ID] = 0;
   else if( name == OVERRIDE_OP_MOD ) m_overrides[OVERRIDE_OP_MOD_ID] = 0;
   else if( name == OVERRIDE_OP_POW ) m_overrides[OVERRIDE_OP_POW_ID] = 0;
   else if( name == OVERRIDE_OP_SHR ) m_overrides[OVERRIDE_OP_SHR_ID] = 0;
   else if( name == OVERRIDE_OP_SHL ) m_overrides[OVERRIDE_OP_SHL_ID] = 0;

   else if( name == OVERRIDE_OP_AADD ) m_overrides[OVERRIDE_OP_AADD_ID] = 0;
   else if( name == OVERRIDE_OP_ASUB ) m_overrides[OVERRIDE_OP_ASUB_ID] = 0;
   else if( name == OVERRIDE_OP_AMUL ) m_overrides[OVERRIDE_OP_AMUL_ID] = 0;
   else if( name == OVERRIDE_OP_ADIV ) m_overrides[OVERRIDE_OP_ADIV_ID] = 0;
   else if( name == OVERRIDE_OP_AMOD ) m_overrides[OVERRIDE_OP_AMOD_ID] = 0;
   else if( name == OVERRIDE_OP_APOW ) m_overrides[OVERRIDE_OP_APOW_ID] = 0;
   else if( name == OVERRIDE_OP_ASHR ) m_overrides[OVERRIDE_OP_ASHR_ID] = 0;
   else if( name == OVERRIDE_OP_ASHL ) m_overrides[OVERRIDE_OP_ASHL_ID] = 0;

   else if( name == OVERRIDE_OP_INC ) m_overrides[OVERRIDE_OP_INC_ID] = 0;
   else if( name == OVERRIDE_OP_DEC ) m_overrides[OVERRIDE_OP_DEC_ID] = 0;
   else if( name == OVERRIDE_OP_INCPOST ) m_overrides[OVERRIDE_OP_INCPOST_ID] = 0;
   else if( name == OVERRIDE_OP_DECPOST ) m_overrides[OVERRIDE_OP_DECPOST_ID] = 0;

   else if( name == OVERRIDE_OP_CALL ) m_overrides[OVERRIDE_OP_CALL_ID] = 0;

   else if( name == OVERRIDE_OP_GETINDEX ) m_overrides[OVERRIDE_OP_GETINDEX_ID] = 0;
   else if( name == OVERRIDE_OP_SETINDEX ) m_overrides[OVERRIDE_OP_SETINDEX_ID] = 0;
   else if( name == OVERRIDE_OP_GETPROP ) m_overrides[OVERRIDE_OP_GETPROP_ID] = 0;
   else if( name == OVERRIDE_OP_SETPROP ) m_overrides[OVERRIDE_OP_SETPROP_ID] = 0;

   else if( name == OVERRIDE_OP_COMPARE ) m_overrides[OVERRIDE_OP_COMPARE_ID] = 0;
   else if( name == OVERRIDE_OP_ISTRUE ) m_overrides[OVERRIDE_OP_ISTRUE_ID] = 0;
   else if( name == OVERRIDE_OP_IN ) m_overrides[OVERRIDE_OP_IN_ID] = 0;
   else if( name == OVERRIDE_OP_TOSTRING ) m_overrides[OVERRIDE_OP_TOSTRING_ID] = 0;
   else if( name == OVERRIDE_OP_ITER ) m_overrides[OVERRIDE_OP_ITER_ID] = 0;
   else if( name == OVERRIDE_OP_UNKMSG ) m_overrides[OVERRIDE_OP_UNKMSG_ID] = 0;

#if OVERRIDE_OP_UNKMSG_ID + 1 != OVERRIDE_OP_COUNT
#error "You forgot to update the operator overrides in MultiClass::checkRemoveOverride"
#endif
}


bool ClassMulti::getOverride( void* self, int op, Class*& cls, void*& udata ) const
{
   Property* override = m_overrides[op];

   if( override != 0 && override->m_itemId >= 0 )
   {
      ItemArray& data = *static_cast<ItemArray*>(self);
      fassert( data.length() > (length_t) override->m_itemId );
      data[override->m_itemId].forceClassInst(cls, udata);
      return true;
   }

   return false;
}


inline bool ClassMulti::inl_get_override( void* self, int op, Class*& cls, void*& udata ) const
{
   Property* override = m_overrides[op];

   if( override != 0 && override->m_itemId >= 0 )
   {
      ItemArray& data = *static_cast<ItemArray*>(self);
      fassert( data.length() > (length_t) override->m_itemId );
      data[override->m_itemId].forceClassInst(cls, udata);
      return true;
   }

   return false;
}


void ClassMulti::op_neg( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override(  self, OVERRIDE_OP_NEG_ID, cls, udata ) )
   {
      cls->op_neg( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_NEG) );
   }
}


void ClassMulti::op_add( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_ADD_ID, cls, udata ) )
   {
      cls->op_add( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_ADD) );
   }
}


void ClassMulti::op_sub( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_SUB_ID, cls, udata ) )
   {
      cls->op_sub( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_SUB) );
   }
}


void ClassMulti::op_mul( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_MUL_ID, cls, udata ) )
   {
      cls->op_mul( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_MUL) );
   }
}


void ClassMulti::op_div( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_DIV_ID, cls, udata ) )
   {
      cls->op_div( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_DIV) );
   }
}

void ClassMulti::op_mod( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_MOD_ID, cls, udata ) )
   {
      cls->op_mod( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_MOD) );
   }
}


void ClassMulti::op_pow( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_POW_ID, cls, udata ) )
   {
      cls->op_pow( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_POW) );
   }
}

void ClassMulti::op_shr( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_SHR_ID, cls, udata ) )
   {
      cls->op_pow( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_SHR) );
   }
}

void ClassMulti::op_shl( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_SHL_ID, cls, udata ) )
   {
      cls->op_pow( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_SHL) );
   }
}


void ClassMulti::op_aadd( VMContext* ctx, void* self) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_AADD_ID, cls, udata ) )
   {
      cls->op_aadd( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_AADD) );
   }
}


void ClassMulti::op_asub( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_ASUB_ID, cls, udata ) )
   {
      cls->op_asub( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_ASUB) );
   }
}


void ClassMulti::op_amul( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_AMUL_ID, cls, udata ) )
   {
      cls->op_amul( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_AMUL) );
   }
}


void ClassMulti::op_adiv( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_DIV_ID, cls, udata ) )
   {
      cls->op_adiv( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_DIV) );
   }
}


void ClassMulti::op_amod( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_AMOD_ID, cls, udata ) )
   {
      cls->op_amod( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam( e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_AMOD) );
   }
}


void ClassMulti::op_apow( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_APOW_ID, cls, udata ) )
   {
      cls->op_apow( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_APOW) );
   }
}


void ClassMulti::op_ashr( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_ASHR_ID, cls, udata ) )
   {
      cls->op_apow( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_ASHR) );
   }
}


void ClassMulti::op_ashl( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_ASHL_ID, cls, udata ) )
   {
      cls->op_apow( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_ASHL) );
   }
}



void ClassMulti::op_inc( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_INC_ID, cls, udata ) )
   {
      cls->op_inc( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_INC) );
   }
}


void ClassMulti::op_dec( VMContext* ctx, void* self) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_DEC_ID, cls, udata ) )
   {
      cls->op_dec( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_DEC) );
   }
}


void ClassMulti::op_incpost( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_INCPOST_ID, cls, udata ) )
   {
      cls->op_incpost( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_INCPOST) );
   }
}


void ClassMulti::op_decpost( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_DECPOST_ID, cls, udata ) )
   {
      cls->op_decpost( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_DECPOST) );
   }
}


void ClassMulti::op_getIndex( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_GETINDEX_ID, cls, udata ) )
   {
      cls->op_getIndex( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_GETINDEX) );
   }
}


void ClassMulti::op_setIndex( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_SETINDEX_ID, cls, udata ) )
   {
      cls->op_setIndex( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_SETINDEX) );
   }
}


bool ClassMulti::getProperty( const String& name, void* instance, Item& target ) const
{
   Private_base::PropMap::const_iterator iter = _p_base->m_props.find( name );
   if( iter != _p_base->m_props.end() )
   {
      const Property& prop = iter->second;
      ItemArray* ia = static_cast<ItemArray*>(instance);

      // if < 0 it's a class.
      if( prop.m_itemId < 0 )
      {
         // so, turn the thing in the "self" of the class.
         target.copyFromRemote(ia->at(-prop.m_itemId));
         return true;
      }
      else
      {
         Class* cls;
         void* udata;
         ia->at(prop.m_itemId).forceClassInst( cls, udata );
         return cls->getProperty( name, udata, target );
      }
   }

   return false;
}


bool ClassMulti::setProperty( const String& name, void* instance, const Item& target ) const
{
   Private_base::PropMap::const_iterator iter = _p_base->m_props.find( name );
   if( iter != _p_base->m_props.end() )
   {
      const Property& prop = iter->second;
      ItemArray* ia = static_cast<ItemArray*>(instance);

      // if < 0 it's a class.
      if( prop.m_itemId < 0 )
      {
         // so, turn the thing in the "self" of the class.
         ia->at(-prop.m_itemId).copyFromLocal(target);
         return true;
      }
      else
      {
         Class* cls;
         void* udata;
         ia->at(prop.m_itemId).forceClassInst( cls, udata );
         return cls->setProperty( name, udata, target );
      }
   }

   return false;
}


void ClassMulti::op_getProperty( VMContext* ctx, void* self, const String& propName ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_GETPROP_ID, cls, udata ) )
   {
      cls->op_getProperty( ctx, udata, propName );
   }
   else
   {
      Private_base::PropMap::const_iterator iter = _p_base->m_props.find( propName );
      if( iter != _p_base->m_props.end() )
      {
         const Property& prop = iter->second;
         ItemArray* ia = static_cast<ItemArray*>(self);

         // if < 0 it's a class.
         if( prop.m_itemId < 0 )
         {
            // so, turn the thing in the "self" of the class.
            ctx->topData() = ia->at(-prop.m_itemId);
         }
         else
         {
            Class* cls;
            void* udata;
            ia->at(prop.m_itemId).forceClassInst( cls, udata );
            cls->op_getProperty( ctx, udata, propName );
         }
      }
      else
      {
         Class::op_getProperty( ctx, self, propName );
      }
   }
}


void ClassMulti::op_setProperty( VMContext* ctx, void* self, const String& propName ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_SETPROP_ID, cls, udata ) )
   {
      cls->op_setProperty( ctx, udata, propName );
   }
   else
   {
      Private_base::PropMap::const_iterator iter = _p_base->m_props.find( propName );
      if( iter != _p_base->m_props.end() )
      {
         const Property& prop = iter->second;
         ItemArray* ia = static_cast<ItemArray*>(self);

         // if < 0 it's a class.
         if( prop.m_itemId < 0 )
         {
            // you can't overwrite a base class.
            throw new AccessError( ErrorParam(e_prop_ro, __LINE__, SRC ).extra(propName) );
         }
         else
         {
            Class* cls;
            void* udata;
            ia->at(prop.m_itemId).forceClassInst( cls, udata );
            cls->op_setProperty( ctx, udata, propName );
         }
      }
      else
      {
         throw new AccessError( ErrorParam(e_prop_acc, __LINE__, SRC ).extra(propName) );
      }
   }
}


void ClassMulti::op_compare( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_COMPARE_ID, cls, udata ) )
   {
      cls->op_compare( ctx, udata );
   }
   else
   {
      // we don't need the self object.
      ctx->popData();
      const Item& crand = ctx->topData();
      if( crand.type() == typeID() )
      {
         // we're all object. Order by ptr.
         ctx->topData() = (int64)(self > crand.asInst() ? 1 : (self < crand.asInst() ? -1 : 0));
      }
      else
      {
         // order by type
         ctx->topData() = (int64)( typeID() - crand.type() );
      }
   }
}


void ClassMulti::op_isTrue( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_ISTRUE_ID, cls, udata ) )
   {
      cls->op_isTrue( ctx, udata );
   }
   else
   {
      // objects are always true.
      ctx->topData() = true;
   }
}


void ClassMulti::op_in( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_IN_ID, cls, udata ) )
   {
      cls->op_in( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_IN) );
   }
}


void ClassMulti::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_CALL_ID, cls, udata ) )
   {
      cls->op_call( ctx, paramCount, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_CALL) );
   }
}


void ClassMulti::op_toString( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_CALL_ID, cls, udata ) )
   {
      cls->op_toString( ctx, udata );
   }
   else
   {
      String* str = new String("Instance of ");
      str->append( name() );
      ctx->topData() = FALCON_GC_STORE( str->handler(), str );
   }
}

void ClassMulti::op_iter( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_ITER_ID, cls, udata ) )
   {
      cls->op_iter( ctx, udata );
   }
   else
   {
      Class::op_iter( ctx, self );
   }
}

void ClassMulti::op_next( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( inl_get_override( self, OVERRIDE_OP_NEXT_ID, cls, udata ) )
   {
      cls->op_next( ctx, udata );
   }
   else
   {
      Class::op_next( ctx, self );
   }
}

}

/* end of multiclass.cpp */
