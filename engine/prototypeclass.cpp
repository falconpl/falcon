/*
   FALCON - The Falcon Programming Language.
   FILE: prototypeclass.cpp

   Class holding more user-type classes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 06:35:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/prototypeclass.cpp"

#include <falcon/prototypeclass.h>
#include <falcon/flexydict.h>
#include <falcon/vmcontext.h>
#include <falcon/stderrors.h>
#include <falcon/stdhandlers.h>

#include <map>
#include <vector>
#include <cstring>

#include <falcon/ov_names.h>

namespace Falcon
{

PrototypeClass::PrototypeClass():
   FlexyClass( "Prototype", FLC_CLASS_ID_PROTO )
{
}


PrototypeClass::~PrototypeClass()
{
}


bool PrototypeClass::hasProperty( void* self, const String& prop ) const
{
   FlexyDict* fd = static_cast<FlexyDict*>(self);
   if( fd->hasProperty( prop) )
   {
      return true;
   }

   // nope -- search across bases.
   for( length_t i = 0; i < fd->base().length(); ++i )
   {
      Class* cls;
      void* data;
      fd->base()[i].forceClassInst( cls, data );
      if( cls->hasProperty( data, prop ) )
      {
         return true;
      }
   }
   
   return false;
}

//=========================================================
// Operators.
//

bool PrototypeClass::op_init( VMContext* ctx, void* instance, int32 params ) const
{
   FlexyDict *value = static_cast<FlexyDict*>(instance);
   value->setBaseType(true);

   // we must create the prototype with the given bases.
   if( params > 0 )
   {
      ItemArray& base = value->base();
      base.reserve( params );

      Item* result = ctx->opcodeParams(params);
      Item* end = result + params;
      while( result < end )
      {
         base.append( *result );
         ++result;
      }
   }

   return false;
}


void PrototypeClass::op_call( VMContext* ctx, int32 paramCount, void* instance ) const
{
   FlexyDict *self = static_cast<FlexyDict*>(instance);
   
   if( ! self->isBaseType() )
   {

      FlexyDict* self = static_cast<FlexyDict*>(instance);
      Item* override = 0;

      if( self->meta() )
      {
         override = self->meta()->find(OVERRIDE_OP_CALL);
      }

      if( override != 0 && override->isFunction() )
      {
         ctx->callInternal( override->asFunction(), paramCount, Item( this, self ) );
      }
      else
      {
         FlexyClass::op_call( ctx, paramCount, instance );
      }

      return;
   }
   
   FlexyDict* child = new FlexyDict;
   child->base().resize(1);
   child->base()[0].setUser(this, self);

   Item* iNewSelf = ctx->opcodeParams(paramCount + 1);
   iNewSelf->setUser(FALCON_GC_STORE(this, child) );
   // do automatic meta-init transfer.
   if( self->meta() != 0 )
   {
      child->meta( self->meta() );
   }

   Item* init = self->find(FALCON_PROTOTYPE_PROPERTY_OVERRIDE_INIT);

   if( init != 0 )
   {
      if( init->isMethod() )
      {
         ctx->callInternal( init->asMethodFunction(), paramCount, *init );
      }
      else if( init->isFunction() )
      {
         ctx->callInternal( init->asFunction(), paramCount, *iNewSelf );
      }

      return;
   }

   ctx->popData(paramCount);
   // return the new self object that we have already created on top of the stack.
}


void PrototypeClass::op_getProperty( VMContext* ctx, void* self, const String& prop ) const
{
   FlexyDict* fd = static_cast<FlexyDict*>(self);

   // check the special property _base
   if( prop == FALCON_PROTOTYPE_PROPERTY_OVERRIDE_BASE )
   {
      ctx->topData() = Item(fd->base().handler(), &fd->base());
   }   
   else if( prop == FALCON_PROTOTYPE_PROPERTY_OVERRIDE_PROTOTYPE )
   {
      if( fd->base().length() > 0 )
      {
         ctx->topData() = fd->base()[0];
      }
      else {
         ctx->topData().setNil();
      }
   }
   else if( prop == FALCON_PROTOTYPE_PROPERTY_OVERRIDE_ISBASE )
   {
      ctx->topData().setBoolean( fd->isBaseType() );
   }
   else if( prop == FALCON_PROTOTYPE_PROPERTY_OVERRIDE_META )
   {
      if( fd->meta() ) {
         ctx->topData().setUser(this, fd->meta() );
      }
      else {
         ctx->topData().setNil();
      }
   }
   else 
   {
      if(fd->meta() != 0 && prop.getCharAt(0) != '_' )
      {
         Item* meta = fd->meta()->find( OVERRIDE_OP_GETPROP );
         if( meta != 0 )
         {
            if( meta->isFunction() )
            {
               ctx->pushData(FALCON_GC_HANDLE(new String(prop)));
               ctx->callInternal(meta->asFunction(), 1, Item(this, fd) );
               return;
            }
            else if( meta->isMethod() )
            {
               ctx->pushData(FALCON_GC_HANDLE(new String(prop)));
               ctx->callInternal(meta->asMethodFunction(), 1, *meta );
               return;
            }
         }
      }

      Item* item = fd->find( prop );
      if( item != 0 )
      {
         Item& result = *item;
         if( result.isMethod() && result.asClass() == this )
         {
            result.content.data.ptr.pInst = self;
            ctx->topData() = result;
         }
         else if( result.isFunction() ) {
            ctx->topData().setUser(this, self);
            ctx->topData().methodize(result.asFunction());
         }
         else {
            ctx->topData() = result;
         }

         return;
      }
      else
      {
         // nope -- search across bases.
         const ItemArray& base = fd->base();
         for( length_t i = 0; i < base.length(); ++i )
         {
            Class* cls;
            void* data;
            base[i].forceClassInst( cls, data );
            if( cls->hasProperty( data, prop ) )
            {
               long depth = ctx->codeDepth();
               ctx->pushCode( &m_stepGetPropertyNext );

               ctx->pushData( base[i] ); // getProperty wants it and eats it away
               cls->op_getProperty( ctx, data, prop );
               if( ctx->wentDeepSized(depth) )
               {
                  return;
               }

               Item result = ctx->topData();
               ctx->popData();
               if( result.isMethod() && result.asClass() == this )
               {
                  result.content.data.ptr.pInst = self;
                  ctx->topData() = result;
               }
               else if( result.isFunction() ) {
                  ctx->topData().setUser(this, self);
                  ctx->topData().methodize(result.asFunction());
               }
               else {
                  ctx->topData() = result;
               }

               // and we eat ourselves in the stack with our property
               ctx->popCode();
               return;
            }
         }
      }
      
      // fall back to the starndard system
      Class::op_getProperty( ctx, self, prop );
   }
}


void PrototypeClass::PStepGetPropertyNext::apply_( const PStep*, VMContext* ctx )
{
   static Class* protoClass = Engine::handlers()->protoClass();

   Item result = ctx->topData();
   ctx->popData();
   if( result.isMethod() && result.asClass() == protoClass )
   {
      result.content.data.ptr.pInst = ctx->topData().asInst();
      ctx->topData() = result;
   }
   else if( result.isFunction() ) {
      ctx->topData().setUser(protoClass, ctx->topData().asInst());
      ctx->topData().methodize(result.asFunction());
   }
   else {
      ctx->topData() = result;
   }

   ctx->popCode();
}


void PrototypeClass::op_setProperty( VMContext* ctx, void* self, const String& prop ) const
{
   FlexyDict* fd = static_cast<FlexyDict*>(self);

   // check the special property _base
   if( prop == FALCON_PROTOTYPE_PROPERTY_OVERRIDE_BASE )
   {
      ctx->popData();
      Item& value = ctx->topData();
      if( value.isArray() )
      {
         fd->base().resize(0);
         fd->base().merge( *value.asArray() );
      }
      else
      {
         fd->base().resize(1);
         fd->base()[0].copyInterlocked(value);
      }
   }
   else if( prop == FALCON_PROTOTYPE_PROPERTY_OVERRIDE_PROTOTYPE )
   {
      ctx->popData();
      Item& value = ctx->topData();
      
      if( fd->base().length() < 1 )
      {
         fd->base().resize(1);
      }

      fd->base()[0] = value;
   }
   else if( prop == FALCON_PROTOTYPE_PROPERTY_OVERRIDE_ISBASE )
   {
      ctx->popData();
      Item& value = ctx->topData();      
      fd->setBaseType( value.isTrue() );
   }
   else if ( prop == FALCON_PROTOTYPE_PROPERTY_OVERRIDE_META )
   {
      ctx->popData();
      Item& value = ctx->topData();

      // the m_meta is just a proxy.
      if( value.asClass() != this )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_type, .extra("Prototype") );
      }

      // save it in the dictionary so that we don't have to gc or flatten it separately
      FlexyDict* m = static_cast<FlexyDict*>(value.asInst());
      fd->meta( m );
   }
   else
   {
      if(fd->meta() != 0 && prop.getCharAt(0) != '_' )
      {
         Item* meta = fd->meta()->find( OVERRIDE_OP_SETPROP );
         if( meta != 0 )
         {
            if( meta->isFunction() )
            {
               Item temp = ctx->opcodeParam(1);
               ctx->topData() = FALCON_GC_HANDLE(new String(prop));
               ctx->pushData(temp);
               ctx->callInternal(meta->asFunction(), 2, Item(this, fd) );
               return;
            }
            else if( meta->isMethod() )
            {
               Item temp = ctx->opcodeParam(1);
               ctx->topData() = FALCON_GC_HANDLE(new String(prop));
               ctx->pushData(temp);
               ctx->callInternal(meta->asMethodFunction(), 2, *meta );
               return;
            }
         }
      }

      FlexyClass::op_setProperty( ctx, self, prop );
   }
}


void PrototypeClass::describe( void* instance, String& target, int depth, int maxlen) const
{
   String instName;
   
   FlexyDict* fd = static_cast<FlexyDict*>(instance);
   ItemArray& base = fd->base();
   if( base.length() > 0 )
   {
      Class* cls;
      void* inst;
      base[0].forceClassInst(cls, inst);
      if( cls == this )
      {
         Item* iname = static_cast<FlexyDict*>(inst)->find("_name");
         if( iname != 0 && iname->isString() )
         {
            instName = *iname->asString();
         }
         else {
            instName = name();
         }
      }
      else {
         instName = cls->name();
      }
   }
   else
   {
      instName = name();
   }
   
   target.size(0);
   target += instName;
   if( fd->isBaseType() ) {
      target += "*";
   }

   
   if( depth != 0 )
   {
      String tgt;
      fd->describe( tgt, depth, maxlen );
      target += "{" + tgt + "}";
   }
   else
   {
      target += "{...}";
   }  
}


inline bool PrototypeClass::callOverride( VMContext* ctx, FlexyDict* self, const String& opName, int count ) const
{
   if( self->meta() != 0 )
   {
      Item* override = self->meta()->find(opName);

      if( override != 0 )
      {
         if ( override->isFunction() )
         {
            // 1 parameter == second; which will be popped away,
            // while first == self will be substituted with the return value.
            ctx->callInternal( override->asFunction(), count, Item( this, self ) );
            return true;
         }
         else if ( override->isMethod() )
         {
            ctx->callInternal( override->asMethodFunction(), count, *override );
            return true;
         }

         // doing other call controls has no meaning.
      }
   }

   return false;
}


inline void PrototypeClass::override_unary( VMContext* ctx, void* instance, const String& opName ) const
{
   FlexyDict* self = static_cast<FlexyDict*>(instance);
   if( callOverride( ctx, self, opName, 0 ) )
   {
      return;
   }

   throw new OperandError( ErrorParam(e_invop, __LINE__, SRC )
            .extra(opName)
            .origin( ErrorParam::e_orig_vm) );
}


inline void PrototypeClass::override_binary(  VMContext* ctx, void* instance, const String& opName ) const
{
   FlexyDict* self = static_cast<FlexyDict*>(instance);
   if( callOverride( ctx, self, opName, 1 ) )
   {
      return;
   }

   throw new OperandError( ErrorParam(e_invop, __LINE__, SRC )
            .extra(opName)
            .origin( ErrorParam::e_orig_vm) );
}


void PrototypeClass::op_neg( VMContext* ctx, void* self ) const
{
   override_unary( ctx, self, OVERRIDE_OP_NEG );
}


void PrototypeClass::op_add( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_ADD );
}


void PrototypeClass::op_sub( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_SUB );
}


void PrototypeClass::op_mul( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_MUL );
}


void PrototypeClass::op_div( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_DIV );
}

void PrototypeClass::op_mod( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_MOD );
}


void PrototypeClass::op_pow( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_POW );
}

void PrototypeClass::op_shr( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_SHR );
}

void PrototypeClass::op_shl( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_SHL );
}


void PrototypeClass::op_aadd( VMContext* ctx, void* self) const
{
   override_binary( ctx, self, OVERRIDE_OP_AADD );
}


void PrototypeClass::op_asub( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_ASUB );
}


void PrototypeClass::op_amul( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_AMUL );
}


void PrototypeClass::op_adiv( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_ADIV );
}


void PrototypeClass::op_amod( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_AMOD );
}


void PrototypeClass::op_apow( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_APOW );
}

void PrototypeClass::op_ashr( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_ASHR );
}

void PrototypeClass::op_ashl( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_ASHL );
}

void PrototypeClass::op_inc( VMContext* ctx, void* self ) const
{
   override_unary( ctx, self, OVERRIDE_OP_INC );
}


void PrototypeClass::op_dec( VMContext* ctx, void* self) const
{
   override_unary( ctx, self, OVERRIDE_OP_DEC );
}


void PrototypeClass::op_incpost( VMContext* ctx, void* self ) const
{
   override_unary( ctx, self, OVERRIDE_OP_INCPOST );
}


void PrototypeClass::op_decpost( VMContext* ctx, void* self ) const
{
   override_unary( ctx, self, OVERRIDE_OP_DECPOST );
}


void PrototypeClass::op_getIndex( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_GETINDEX );
}


void PrototypeClass::op_setIndex( VMContext* ctx, void* instance ) const
{
   FlexyDict* self = static_cast<FlexyDict*>(instance);

   if( self->meta() != 0 )
   {
      Item* override = self->meta()->find(OVERRIDE_OP_SETINDEX);
      if( override != 0  && override->isFunction() )
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
         ctx->callInternal( override->asFunction(), 2, Item(this, self) );
      }

      return;
   }

   throw new OperandError( ErrorParam(__LINE__, e_invop )
            .extra(OVERRIDE_OP_SETINDEX)
            .origin( ErrorParam::e_orig_vm) );
}


void PrototypeClass::op_compare( VMContext* ctx, void* instance ) const
{
   FlexyDict* self = static_cast<FlexyDict*>(instance);
   Item* override = 0;
   if( self->meta() != 0 )
   {
      override = self->meta()->find(OVERRIDE_OP_COMPARE);
   }

   if( override && override->isFunction() )
   {
      // call will remove the extra parameter...
      Item iSelf( this, self );
      // remove "self" from the stack..
      ctx->popData();
      ctx->callInternal( override->asFunction(), 1, iSelf );
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


void PrototypeClass::op_isTrue( VMContext* ctx, void* instance ) const
{
   FlexyDict* self = static_cast<FlexyDict*>(instance);
   Item* override = 0;

   if( self->meta() != 0 )
   {
      override = self->meta()->find(OVERRIDE_OP_INCPOST);
   }

   if( override != 0 && override->isFunction() )
   {
      // use the instance we know, as first can be moved away.
      ctx->callInternal( override->asFunction(), 0, ctx->topData() );
   }
   else
   {
      // instances are always true.
      ctx->topData().setBoolean(true);
   }
}


void PrototypeClass::op_in( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_IN );
}


void PrototypeClass::op_provides( VMContext* ctx, void* instance, const String& propName ) const
{
   FlexyDict* self = static_cast<FlexyDict*>(instance);
   Item* override = 0;

   if( self->meta() != 0 )
   {
      override = self->meta()->find(OVERRIDE_OP_PROVIDES);
   }

   if( override != 0 && override->isFunction() )
   {
      Item i_self( this, self );
      ctx->pushData( FALCON_GC_HANDLE(new String(propName)) );
      ctx->callInternal( override->asFunction(), 1, i_self );
   }
   else if( override != 0 && override->isMethod() )
   {
      ctx->pushData( FALCON_GC_HANDLE(new String(propName)) );
      ctx->callInternal( override->asMethodFunction(), 1, *override);
   }
   else
   {
      ctx->topData().setBoolean( hasProperty( self, propName ) );
   }
}


void PrototypeClass::op_toString( VMContext* ctx, void* instance ) const
{
   FlexyDict* self = static_cast<FlexyDict*>(instance);
   Item* override = 0;

   if( self->meta() != 0 )
   {
      override = self->meta()->find(OVERRIDE_OP_TOSTRING);
   }

   if( override != 0 && override->isFunction() )
   {
      ctx->callInternal( override->asFunction(), 0, Item( this, self ) );
   }
   else
   {
      String* str = new String("Instance of ");
      str->append( name() );
      ctx->topData() = FALCON_GC_HANDLE(str);
   }
}

void PrototypeClass::op_iter( VMContext* ctx, void* self ) const
{
   ctx->addSpace(1);
   ctx->opcodeParam(0) = ctx->opcodeParam(1);

   override_unary( ctx, self, OVERRIDE_OP_ITER );
}

void PrototypeClass::op_next( VMContext* ctx, void* self ) const
{
   ctx->addSpace(2);
   ctx->opcodeParam(0) = ctx->opcodeParam(2);
   ctx->opcodeParam(1) = ctx->opcodeParam(3);

   override_binary( ctx, self, OVERRIDE_OP_NEXT );
}


}

/* end of prototypeclass.cpp */

