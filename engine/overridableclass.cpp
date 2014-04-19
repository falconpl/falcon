/*
   FALCON - The Falcon Programming Language.
   FILE: overridableclass.h

   Base abstract class for classes providing an override system.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 18 Jul 2011 01:55:55 +0200

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
#include <falcon/itemid.h>
#include <falcon/stderrors.h>

#include <cstring>

namespace Falcon
{

OverridableClass::OverridableClass( const String& name ):
   ClassMantra(name, FLC_CLASS_ID_PROTO )
{
   m_overrides = new Function*[OVERRIDE_OP_COUNT_ID];
   memset( m_overrides, 0, sizeof( Function* ) * OVERRIDE_OP_COUNT_ID );
}


OverridableClass::OverridableClass( const String& name, int64 tid ):
   ClassMantra( name, tid )
{
   m_overrides = new Function*[OVERRIDE_OP_COUNT_ID];
   memset( m_overrides, 0, sizeof( Function* ) * OVERRIDE_OP_COUNT_ID );
}


OverridableClass::~OverridableClass()
{
      delete[] m_overrides;
}

void OverridableClass::op_summon_failing( VMContext* ctx, void* instance, const String& message, int32 pCount ) const
{
   if ( m_overrides[OVERRIDE_OP_UNKMSG_ID] != 0 )
   {
      if( pCount == 0 )
      {
         ctx->pushData(FALCON_GC_HANDLE(new String(message)));
      }
      else
      {
         Item i_msg = FALCON_GC_HANDLE(new String(message));
         ctx->insertData(pCount, &i_msg, 1, 0);
      }
      ctx->callInternal( m_overrides[OVERRIDE_OP_UNKMSG_ID], pCount+1 );
   }
   else {
      Class::op_summon_failing(ctx, instance, message, pCount);
   }
}


inline void OverridableClass::override_unary( VMContext* ctx, void* self, int op, const String& opName ) const
{
   Function* override = m_overrides[op];

   // TODO -- use pre-caching of the desired method
   if( override != 0 )
   {
      ctx->callInternal( override, 0, Item( this, self ) );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC )
               .extra(opName)
               .origin( ErrorParam::e_orig_vm) );   }
}


inline void OverridableClass::override_binary( VMContext* ctx, void* self, int op, const String& opName ) const
{
   Function* override = m_overrides[op];

   if( override )
   {
      // we don't need the self in the stack.

      // 1 parameter == second; which will be popped away,
      // while first == self will be substituted with the return value.
      ctx->callInternal( override, 1, Item( this, self ) );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC )
         .extra(opName)
         .origin( ErrorParam::e_orig_vm)
         );
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
   else if( name == OVERRIDE_OP_SHR ) m_overrides[OVERRIDE_OP_SHR_ID] = mth;
   else if( name == OVERRIDE_OP_SHL ) m_overrides[OVERRIDE_OP_SHL_ID] = mth;

   else if( name == OVERRIDE_OP_AADD ) m_overrides[OVERRIDE_OP_AADD_ID] = mth;
   else if( name == OVERRIDE_OP_ASUB ) m_overrides[OVERRIDE_OP_ASUB_ID] = mth;
   else if( name == OVERRIDE_OP_AMUL ) m_overrides[OVERRIDE_OP_AMUL_ID] = mth;
   else if( name == OVERRIDE_OP_ADIV ) m_overrides[OVERRIDE_OP_ADIV_ID] = mth;
   else if( name == OVERRIDE_OP_AMOD ) m_overrides[OVERRIDE_OP_AMOD_ID] = mth;
   else if( name == OVERRIDE_OP_APOW ) m_overrides[OVERRIDE_OP_APOW_ID] = mth;
   else if( name == OVERRIDE_OP_ASHR ) m_overrides[OVERRIDE_OP_ASHR_ID] = mth;
   else if( name == OVERRIDE_OP_ASHL ) m_overrides[OVERRIDE_OP_ASHL_ID] = mth;

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
   else if( name == OVERRIDE_OP_ITER ) m_overrides[OVERRIDE_OP_ITER_ID] = mth;
   else if( name == OVERRIDE_OP_NEXT ) m_overrides[OVERRIDE_OP_NEXT_ID] = mth;
   else if( name == OVERRIDE_OP_UNKMSG ) m_overrides[OVERRIDE_OP_UNKMSG_ID] = mth;

#if OVERRIDE_OP_UNKMSG_ID + 1 != OVERRIDE_OP_COUNT
#error "You forgot to update the operator overrides in OverridableClass::overrideAddMethod"
#endif
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
   else if( name == OVERRIDE_OP_PROVIDES ) m_overrides[OVERRIDE_OP_PROVIDES_ID] = 0;
   else if( name == OVERRIDE_OP_TOSTRING ) m_overrides[OVERRIDE_OP_TOSTRING_ID] = 0;
   else if( name == OVERRIDE_OP_ITER ) m_overrides[OVERRIDE_OP_ITER_ID] = 0;
   else if( name == OVERRIDE_OP_NEXT ) m_overrides[OVERRIDE_OP_NEXT_ID] = 0;
   else if( name == OVERRIDE_OP_UNKMSG ) m_overrides[OVERRIDE_OP_UNKMSG_ID] = 0;

#if OVERRIDE_OP_UNKMSG_ID + 1 != OVERRIDE_OP_COUNT
#error "You forgot to update the operator overrides in OverridableClass::overrideRemoveMethod"
#endif

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

void OverridableClass::op_shr( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_SHR_ID, OVERRIDE_OP_SHR );
}

void OverridableClass::op_shl( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_SHL_ID, OVERRIDE_OP_SHL );
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

void OverridableClass::op_ashr( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_ASHR_ID, OVERRIDE_OP_ASHR );
}

void OverridableClass::op_ashl( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_ASHL_ID, OVERRIDE_OP_ASHL );
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
      Item* params = ctx->opcodeParams(3);
      Item value = params[0];
      Item iself = params[1];
      Item nth = params[2];
      params[0] = iself;
      params[1] = nth;
      params[2] = value;

      // Two parameters (second and third) will be popped,
      //  and first will be turned in the result.
      ctx->callInternal( override, 2, iself );
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
      Item i_first( this, self );
      ctx->popData();
      // I prefer to go safe and push a new string here.
      ctx->pushData( FALCON_GC_HANDLE(new String(propName)) );

      // use the instance we know, as first can be moved away.
      ctx->callInternal( override, 1, i_first );

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
      Item iSelf( this, self );
      // remove "self" from the stack..
      ctx->popData();
      // exchange the property name and the data,
      // as setProperty wants the propname first.
      Item i_data = ctx->topData();
      ctx->topData() = FALCON_GC_HANDLE(new String(propName));
      ctx->pushData( i_data );

      // Don't mangle the stack, we have to change it.
      ctx->callInternal( override, 2, iSelf );

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
      Item iSelf( this, self );
      // remove "self" from the stack..
      ctx->popData();
      ctx->callInternal( override, 1, iSelf );
   }
   else
   {
      // we don't need the self object.
      Class::op_compare(ctx, self);
   }
}


void OverridableClass::op_isTrue( VMContext* ctx, void* ) const
{
   Function* override = m_overrides[OVERRIDE_OP_ISTRUE_ID];

   if( override != 0 )
   {
      // use the instance we know, as first can be moved away.
      ctx->callInternal( override, 0, ctx->topData() );
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
      Item i_self( this, self );
      ctx->topData() = FALCON_GC_HANDLE(new String(propName));
      ctx->callInternal( override, 1, i_self );
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
      //ctx->popData();
      ctx->callInternal( override, paramCount, Item( this, self ) );
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
      ctx->callInternal( override, 0, Item( this, self ) );
   }
   else
   {
      String* str = new String("Instance of ");
      str->append( name() );
      ctx->topData() = FALCON_GC_HANDLE(str);
   }
}

void OverridableClass::op_iter( VMContext* ctx, void* self ) const
{
   ctx->addSpace(1);
   ctx->opcodeParam(0) = ctx->opcodeParam(1);

   override_unary( ctx, self, OVERRIDE_OP_ITER_ID, OVERRIDE_OP_ITER );
}

void OverridableClass::op_next( VMContext* ctx, void* self ) const
{
   ctx->addSpace(2);
   ctx->opcodeParam(0) = ctx->opcodeParam(2);
   ctx->opcodeParam(1) = ctx->opcodeParam(3);
   override_binary( ctx, self, OVERRIDE_OP_NEXT_ID, OVERRIDE_OP_NEXT );
}


}

/* end of overridableclass.cpp */
