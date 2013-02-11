/*
   FALCON - The Falcon Programming Language.
   FILE: class.cpp

   Class definition of a Falcon Class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Jan 2011 15:01:08 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/class.cpp"

#include <falcon/trace.h>
#include <falcon/module.h>
#include <falcon/class.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/bom.h>
#include <falcon/error.h>
#include <falcon/errors/operanderror.h>
#include <falcon/errors/unserializableerror.h>
#include <falcon/errors/accesserror.h>
#include <falcon/errors/accesstypeerror.h>

namespace Falcon {

Class::Class( const String& name ):
   Mantra( name, 0, 0, 0 ),
   m_bIsfalconClass( false ),
   m_bIsErrorClass( false ),
   m_bIsFlatInstance(false),
   m_userFlags(0),
   m_typeID( FLC_ITEM_USER ),
   m_clearPriority( 0 )
{
   m_category = e_c_class;
}

Class::Class( const String& name, int64 tid ):
   Mantra( name, 0, 0, 0 ),
   m_bIsfalconClass( false ),
   m_bIsErrorClass( false ),
   m_bIsFlatInstance(false),
   m_userFlags(0),
   m_typeID( tid ),
   m_clearPriority( 0 )
{
   m_category = e_c_class;
}

Class::Class( const String& name, Module* module, int line, int chr ):
   Mantra( name, module, line, chr ),
   m_bIsfalconClass( false ),
   m_bIsErrorClass( false ),
   m_bIsFlatInstance(false),
   m_userFlags(0),
   m_typeID( FLC_ITEM_USER ),
   m_clearPriority( 0 )
{
   m_category = e_c_class;
}

Class::Class( const String& name, int64 tid, Module* module, int line, int chr ):
   Mantra( name, module, line, chr ),
   m_bIsfalconClass( false ),
   m_bIsErrorClass( false ),
   m_bIsFlatInstance(false),
   m_userFlags(0),
   m_typeID( tid ),
   m_clearPriority( 0 )
{
   m_category = e_c_class;
}


Class::~Class()
{
   TRACE1( "Destroying class %s.%s",
      m_module != 0 ? m_module->name().c_ize() : "<internal>",
      m_name.c_ize() );
}


Class* Class::handler() const
{
   static Class* meta = Engine::instance()->metaClass();
   return meta;
}

Class* Class::getParent( const String& ) const
{
   // normally does nothing
   return 0;
}


bool Class::isDerivedFrom( const Class* cls ) const
{
   return this == cls;
}


void Class::enumerateParents( Class::ClassEnumerator&  ) const
{
   // normally does nothing
}

void* Class::getParentData( Class* parent, void* data ) const
{
   if( parent == this ) return data;
   return 0;
}

int64 Class::occupiedMemory( void* ) const
{
   return 0;
}
 
void Class::store( VMContext*, DataWriter*, void* ) const
{
      throw new UnserializableError(ErrorParam( e_unserializable, __LINE__, __FILE__ )
      .origin( ErrorParam::e_orig_vm )
      .extra(name() + " unsupported store"));
}


void Class::restore( VMContext*, DataReader*) const
{
   throw new UnserializableError(ErrorParam( e_unserializable, __LINE__, __FILE__ )
      .origin( ErrorParam::e_orig_vm )
      .extra(name() + " unsupported restore"));
}


void Class::flatten( VMContext*, ItemArray&, void* ) const
{
   // normally does nothing
}


void Class::unflatten( VMContext*, ItemArray&, void* ) const
{
   // normally does nothing
}


void Class::gcMarkInstance( void*, uint32 ) const
{
   // normally does nothing
}


bool Class::gcCheckInstance( void*, uint32 ) const
{
   return true;
}


void Class::describe( void*, String& target, int, int ) const
{
   target = "<" + name() + "*>";
}


void Class::enumerateProperties( void*, Class::PropertyEnumerator& ) const
{
   // normally does nothing
}

void Class::enumeratePV( void*, Class::PVEnumerator& ) const
{
   // normally does nothing
}


bool Class::hasProperty( void*, const String& ) const
{
   return false;
}


void Class::op_compare( VMContext* ctx, void* self ) const
{
   void* inst;
   Item *op1, *op2;
   
   ctx->operands( op1, op2 );
   
   if( op2->isUser() )
   {
      if( (inst = op2->asInst()) == self )
      {
         ctx->stackResult(2, 0 );
         return;
      }

      if( typeID() > 0 )
      {
         ctx->stackResult(2, (int64)  typeID() - op2->asClass()->typeID() );
         return;
      }
   }

   // we have no information about what an item might be here, but we can
   // order the items by type
   ctx->stackResult(2, (int64) op1->type() - op2->type() );
}


void Class::onInheritanceResolved( ExprInherit* )
{
   // do nothing
}

//=====================================================================
// VM Operator override.
//

bool Class::op_init( VMContext*, void*, int32 ) const
{
   throw FALCON_SIGN_XERROR( OperandError, e_invop, .extra("init") );
}


void Class::op_neg( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("neg") );
}

void Class::op_add( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("add") );
}

void Class::op_sub( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("sub") );
}


void Class::op_mul( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("mul") );
}


void Class::op_div( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("div") );
}


void Class::op_mod( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("mod") );
}


void Class::op_pow( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("pow") );
}

void Class::op_shr( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("shift right (>>)") );
}

void Class::op_shl( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("shift left (<<)") );
}


void Class::op_aadd( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto add (+=)") );
}


void Class::op_asub( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto sub (-=)") );
}


void Class::op_amul( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto amul (-=)") );;
}


void Class::op_adiv( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto div (/=)") );
}


void Class::op_amod( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto mod (%=)") );
}


void Class::op_apow( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto pow (**=)") );
}

void Class::op_ashr( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto shr (>>=)") );
}

void Class::op_ashl( VMContext* ctx, void*  ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto shl (<<=)") );
}


void Class::op_inc( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Pre-inc (++x)") );
}


void Class::op_dec( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Pre-dec (--x)") );
}


void Class::op_incpost( VMContext* ctx, void*) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Post-inc (x++)") );
}


void Class::op_decpost( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Post-dec (x--)") );
}


void Class::op_call( VMContext* ctx, int32 count , void* ) const
{
   ctx->popData(count);
}


void Class::op_getIndex(VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Get index []") );
}


void Class::op_setIndex(VMContext* ctx, void* ) const
{
   // TODO: IS it worth to add more infos about self in the error?
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
                .extra("Set index []=") );
}


void Class::op_getProperty( VMContext* ctx, void* data, const String& property ) const
{
   static BOM* bom = Engine::instance()->getBom();

   // try to find a valid BOM propery.
   BOM::handler handler = bom->get( property );
   if ( handler != 0  )
   {
      handler( ctx, this, data );
   }
   else
   {
      FALCON_RESIGN_XERROR( AccessTypeError, e_prop_acc, ctx,
                   .extra(property) );
   }
}


void Class::op_setProperty( VMContext* ctx, void*, const String& prop ) const
{
   FALCON_RESIGN_XERROR( AccessError, e_prop_acc, ctx,
                .extra(prop) );
}


void Class::op_isTrue( VMContext* ctx, void* ) const
{
   ctx->topData().setBoolean(true);
}


void Class::op_in( VMContext* ctx, void*) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
                .extra("in") );
}

void Class::op_provides( VMContext* ctx, void* instance, const String& propName ) const
{
   ctx->topData().setBoolean( hasProperty(instance, propName ) );
}


void Class::op_toString( VMContext* ctx, void *self ) const
{
   String *descr = new String();
   describe( self, *descr );
   ctx->stackResult(1, FALCON_GC_HANDLE(descr));
}


void Class::op_iter( VMContext* ctx, void* ) const
{
   Item item;
   item.setBreak();
   ctx->pushData(item);
}

void Class::op_next( VMContext* ctx, void* ) const
{
   ctx->topData().setBreak();
}

}

/* end of class.cpp */
