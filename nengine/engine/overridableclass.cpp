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

#define SRC "falcon/overridableclass.cpp"

#include <falcon/overridableclass.h>
#include <falcon/ov_names.h>
// If inlined...
#include <falcon/vmcontext.h>
#include <falcon/function.h>
#include <falcon/operanderror.h>

#include <cstring>

namespace Falcon
{

OverridableClass::OverridableClass( const String& name ):
   Class(name)
{
   m_overrides = new Function*[OVERRIDE_OP_COUNT_ID];
   memset( m_overrides, 0, sizeof( Function* ) * OVERRIDE_OP_COUNT_ID );
}


OverridableClass::OverridableClass( const String& name, int64 tid ):
   Class( name, tid )
{
   m_overrides = new Function*[OVERRIDE_OP_COUNT_ID];
   memset( m_overrides, 0, sizeof( Function* ) * OVERRIDE_OP_COUNT_ID );
}


OverridableClass::~OverridableClass()
{
      delete[] m_overrides;
}


inline void OverridableClass::override_unary( VMContext* ctx, void* self, int op, const String& opName ) const
{
   Function* override = m_overrides[op];

   // TODO -- use pre-caching of the desired method
   if( override != 0 )
   {
      ctx->call( override, 0, Item( this, self, true ), true );
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(opName) );
   }
}


inline void OverridableClass::override_binary( VMContext* ctx, void* self, int op, const String& opName ) const
{
   Function* override = m_overrides[op];

   if( override )
   {
      // we don't need the self in the stack.

      // 1 parameter == second; which will be popped away,
      // while first == self will be substituted with the return value.
      ctx->call( override, 1, Item( this, self, true ), true );
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(opName) );
   }
}


void OverridableClass::overrideAddMethod( const String& name, Function* mth )
{
   // see if the method is an override.
   if( name == OVERRIDE_OP_NEG ) m_overrides[OVERRIDE_OP_NEG_ID] = mth;

   else if( name == OVERRIDE_OP_ADD ) m_overrides[OVERRIDE_OP_ADD_ID] = mth;
   else if( name == OVERRIDE_OP_SUB ) m_overrides[OVERRIDE_OP_SUB_ID] = mth;
   else if( name == OVERRIDE_OP_MUL ) m_overrides[OVERRIDE_OP_MUL_ID] = mth;
   else if( name == OVERRIDE_OP_DIV ) m_overrides[OVERRIDE_OP_DIV_ID] = mth;
   else if( name == OVERRIDE_OP_MOD ) m_overrides[OVERRIDE_OP_MOD_ID] = mth;
   else if( name == OVERRIDE_OP_POW ) m_overrides[OVERRIDE_OP_POW_ID] = mth;

   else if( name == OVERRIDE_OP_AADD ) m_overrides[OVERRIDE_OP_AADD_ID] = mth;
   else if( name == OVERRIDE_OP_ASUB ) m_overrides[OVERRIDE_OP_ASUB_ID] = mth;
   else if( name == OVERRIDE_OP_AMUL ) m_overrides[OVERRIDE_OP_AMUL_ID] = mth;
   else if( name == OVERRIDE_OP_ADIV ) m_overrides[OVERRIDE_OP_ADIV_ID] = mth;
   else if( name == OVERRIDE_OP_AMOD ) m_overrides[OVERRIDE_OP_AMOD_ID] = mth;
   else if( name == OVERRIDE_OP_APOW ) m_overrides[OVERRIDE_OP_APOW_ID] = mth;

   else if( name == OVERRIDE_OP_INC ) m_overrides[OVERRIDE_OP_INC_ID] = mth;
   else if( name == OVERRIDE_OP_DEC ) m_overrides[OVERRIDE_OP_DEC_ID] = mth;
   else if( name == OVERRIDE_OP_INCPOST ) m_overrides[OVERRIDE_OP_INCPOST_ID] = mth;
   else if( name == OVERRIDE_OP_DECPOST ) m_overrides[OVERRIDE_OP_DECPOST_ID] = mth;

   else if( name == OVERRIDE_OP_CALL ) m_overrides[OVERRIDE_OP_CALL_ID] = mth;

   else if( name == OVERRIDE_OP_GETINDEX ) m_overrides[OVERRIDE_OP_GETINDEX_ID] = mth;
   else if( name == OVERRIDE_OP_SETINDEX ) m_overrides[OVERRIDE_OP_SETINDEX_ID] = mth;
   else if( name == OVERRIDE_OP_GETPROP ) m_overrides[OVERRIDE_OP_GETPROP_ID] = mth;
   else if( name == OVERRIDE_OP_SETPROP ) m_overrides[OVERRIDE_OP_SETPROP_ID] = mth;

   else if( name == OVERRIDE_OP_COMPARE ) m_overrides[OVERRIDE_OP_COMPARE_ID] = mth;
   else if( name == OVERRIDE_OP_ISTRUE ) m_overrides[OVERRIDE_OP_ISTRUE_ID] = mth;
   else if( name == OVERRIDE_OP_IN ) m_overrides[OVERRIDE_OP_IN_ID] = mth;
   else if( name == OVERRIDE_OP_PROVIDES ) m_overrides[OVERRIDE_OP_PROVIDES_ID] = mth;
   else if( name == OVERRIDE_OP_TOSTRING ) m_overrides[OVERRIDE_OP_TOSTRING_ID] = mth;
}


void OverridableClass::overrideRemoveMethod( const String& name )
{
   // see if the method is an override.
   if( name == OVERRIDE_OP_NEG ) m_overrides[OVERRIDE_OP_NEG_ID] = 0;

   else if( name == OVERRIDE_OP_ADD ) m_overrides[OVERRIDE_OP_ADD_ID] = 0;
   else if( name == OVERRIDE_OP_SUB ) m_overrides[OVERRIDE_OP_SUB_ID] = 0;
   else if( name == OVERRIDE_OP_MUL ) m_overrides[OVERRIDE_OP_MUL_ID] = 0;
   else if( name == OVERRIDE_OP_DIV ) m_overrides[OVERRIDE_OP_DIV_ID] = 0;
   else if( name == OVERRIDE_OP_MOD ) m_overrides[OVERRIDE_OP_MOD_ID] = 0;
   else if( name == OVERRIDE_OP_POW ) m_overrides[OVERRIDE_OP_POW_ID] = 0;

   else if( name == OVERRIDE_OP_AADD ) m_overrides[OVERRIDE_OP_AADD_ID] = 0;
   else if( name == OVERRIDE_OP_ASUB ) m_overrides[OVERRIDE_OP_ASUB_ID] = 0;
   else if( name == OVERRIDE_OP_AMUL ) m_overrides[OVERRIDE_OP_AMUL_ID] = 0;
   else if( name == OVERRIDE_OP_ADIV ) m_overrides[OVERRIDE_OP_ADIV_ID] = 0;
   else if( name == OVERRIDE_OP_AMOD ) m_overrides[OVERRIDE_OP_AMOD_ID] = 0;
   else if( name == OVERRIDE_OP_APOW ) m_overrides[OVERRIDE_OP_APOW_ID] = 0;

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
   else if( name == OVERRIDE_OP_PROVIDES ) m_overrides[OVERRIDE_OP_PROVIDES_ID] = 0;
   else if( name == OVERRIDE_OP_TOSTRING ) m_overrides[OVERRIDE_OP_TOSTRING_ID] = 0;
}


void OverridableClass::op_neg( VMContext* ctx, void* self ) const
{
   override_unary( ctx, self, OVERRIDE_OP_NEG_ID, OVERRIDE_OP_NEG );
}


void OverridableClass::op_add( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_ADD_ID, OVERRIDE_OP_ADD );
}


void OverridableClass::op_sub( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_SUB_ID, OVERRIDE_OP_SUB );
}


void OverridableClass::op_mul( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_MUL_ID, OVERRIDE_OP_MUL );
}


void OverridableClass::op_div( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_DIV_ID, OVERRIDE_OP_DIV );
}

void OverridableClass::op_mod( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_MOD_ID, OVERRIDE_OP_MOD );
}


void OverridableClass::op_pow( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_POW_ID, OVERRIDE_OP_POW );
}


void OverridableClass::op_aadd( VMContext* ctx, void* self) const
{
   override_binary( ctx, self, OVERRIDE_OP_AADD_ID, OVERRIDE_OP_AADD );
}


void OverridableClass::op_asub( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_ASUB_ID, OVERRIDE_OP_ASUB );
}


void OverridableClass::op_amul( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_AMUL_ID, OVERRIDE_OP_AMUL );
}


void OverridableClass::op_adiv( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_DIV_ID, OVERRIDE_OP_DIV );
}


void OverridableClass::op_amod( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_AMOD_ID, OVERRIDE_OP_AMOD );
}


void OverridableClass::op_apow( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_APOW_ID, OVERRIDE_OP_APOW );
}


void OverridableClass::op_inc( VMContext* ctx, void* self ) const
{
   override_unary( ctx, self, OVERRIDE_OP_INC_ID, OVERRIDE_OP_INC );
}


void OverridableClass::op_dec( VMContext* ctx, void* self) const
{
   override_unary( ctx, self, OVERRIDE_OP_DEC_ID, OVERRIDE_OP_DEC );
}


void OverridableClass::op_incpost( VMContext* ctx, void* self ) const
{
   override_unary( ctx, self, OVERRIDE_OP_INCPOST_ID, OVERRIDE_OP_INCPOST );
}


void OverridableClass::op_decpost( VMContext* ctx, void* self ) const
{
   override_unary( ctx, self, OVERRIDE_OP_DECPOST_ID, OVERRIDE_OP_DECPOST );
}


void OverridableClass::op_getIndex( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_GETINDEX_ID, OVERRIDE_OP_GETINDEX );
}


void OverridableClass::op_setIndex( VMContext* ctx, void* ) const
{
   Function* override = m_overrides[OVERRIDE_OP_SETINDEX_ID];

   if( override != 0 )
   {
      Item* first = ctx->opcodeParams(3);

      // Two parameters (second and third) will be popped,
      //  and first will be turned in the result.
      ctx->call( override, 2, *first, true );
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(OVERRIDE_OP_SETINDEX) );
   }
}


bool OverridableClass::overrideGetProperty( VMContext* ctx, void* self, const String& propName ) const
{
   Function* override = m_overrides[OVERRIDE_OP_GETPROP_ID];

   if( override != 0 )
   {
      // call will destroy the value, that is the parameters
      Item i_first( this, self, true );
      ctx->popData();
      // I prefer to go safe and push a new string here.
      ctx->pushData( (new String(propName))->garbage() );

      // use the instance we know, as first can be moved away.
      ctx->call( override, 1, i_first, true );

      return true;
   }
   else
   {
      return false;
   }
}


bool OverridableClass::overrideSetProperty( VMContext* ctx, void* self, const String& propName ) const
{
   Function* override = m_overrides[OVERRIDE_OP_SETPROP_ID];

   if( override != 0 )
   {
      // call will remove the extra parameter...
      Item iSelf( this, self, true );
      // remove "self" from the stack..
      ctx->popData();
      // exchange the property name and the data,
      // as setProperty wants the propname first.
      Item i_data = ctx->topData();
      ctx->topData() = (new String(propName))->garbage();
      ctx->pushData( i_data );

      // Don't mangle the stack, we have to change it.
      ctx->call( override, 2, iSelf, false );

      return true;
   }
   else
   {
      return false;
   }
}


void OverridableClass::op_compare( VMContext* ctx, void* self ) const
{
   Function* override = m_overrides[OVERRIDE_OP_COMPARE_ID];

   if( override )
   {
      // call will remove the extra parameter...
      Item iSelf( this, self, true );
      // remove "self" from the stack..
      ctx->popData();
      ctx->call( override, 1, iSelf, true );
   }
   else
   {
      // we don't need the self object.
      ctx->popData();
      const Item& crand = ctx->topData();
      if( crand.type() == typeID() )
      {
         // we're all object. Order by ptr.
         ctx->topData() = (int64)
            (self > crand.asInst() ? 1 : (self < crand.asInst() ? -1 : 0));
      }
      else
      {
         // order by type
         ctx->topData() = (int64)( typeID() - crand.type() );
      }
   }
}


void OverridableClass::op_isTrue( VMContext* ctx, void* ) const
{
   Function* override = m_overrides[OVERRIDE_OP_ISTRUE_ID];

   if( override != 0 )
   {
      // use the instance we know, as first can be moved away.
      ctx->call( override, 0, ctx->topData(), true );
   }
   else
   {
      // instances are always true.
      ctx->topData().setBoolean(true);
   }
}


void OverridableClass::op_in( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_IN_ID, OVERRIDE_OP_IN );
}


void OverridableClass::op_provides( VMContext* ctx, void* self, const String& propName ) const
{
   Function* override = m_overrides[OVERRIDE_OP_PROVIDES_ID];

   if( override != 0  )
   {
      Item i_self( this, self, true );
      ctx->topData() = (new String(propName))->garbage();
      ctx->call( override, 1, i_self, true );
   }
   else
   {
      ctx->topData().setBoolean( hasProperty( self, propName ) );
   }
}


void OverridableClass::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   Function* override = m_overrides[OVERRIDE_OP_CALL_ID];

   if( override != 0 )
   {
      ctx->popData();
      ctx->call( override, paramCount, Item( this, self, true ), true );
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(OVERRIDE_OP_CALL) );
   }
}


void OverridableClass::op_toString( VMContext* ctx, void* self ) const
{
   Function* override = m_overrides[OVERRIDE_OP_TOSTRING_ID];

   if( override != 0 )
   {
      ctx->call( override, 0, Item( this, self, true ), true );
   }
   else
   {
      String* str = new String("Instance of ");
      str->append( name() );
      ctx->topData() = str;
   }
}

}

/* end of overridableclass.cpp */
