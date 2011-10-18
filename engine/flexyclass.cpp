/*
   FALCON - The Falcon Programming Language.
   FILE: flexyclass.cpp

   Class handling flexible objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 18 Jul 2011 02:22:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/flexyclass.cpp"

#include <falcon/flexydict.h>
#include <falcon/flexyclass.h>
#include <falcon/item.h>
#include <falcon/itemdict.h>
#include <falcon/vmcontext.h>
#include <falcon/engine.h>
#include <falcon/ov_names.h>

#include <falcon/errors/accesserror.h>
#include <falcon/errors/operanderror.h>


#include <map>


namespace Falcon
{

FlexyClass::FlexyClass():
   Class( "Flexy" )
{
}

FlexyClass::FlexyClass( const String& name ):
   Class( name )
{
}



FlexyClass::~FlexyClass()
{
}


void FlexyClass::dispose( void* self ) const
{
   delete static_cast<FlexyDict*>(self);
}


void* FlexyClass::clone( void* self ) const
{
   return new FlexyDict( *static_cast<FlexyDict*>(self));
}


void FlexyClass::serialize( DataWriter*, void*  ) const
{
   // TODO
}


void* FlexyClass::deserialize( DataReader* ) const
{
   // TODO
   return 0;
}


void* FlexyClass::getParentData( Class* parent, void* data ) const
{
   // if we're the searched class...
   if( parent == this ) return data;
   
   // else, scan our bases, and provide the classe in our bases with their 
   // -- own data.
   FlexyDict* dict = static_cast<FlexyDict*>(data);
   const ItemArray& bases = dict->base();
   for( length_t i = 0; i < bases.length(); ++i )
   {
      // The items in bases are instances, but not necessarily UserData...
      Class* cls;
      void *udata;
      //... so we have to force them as UserData
      bases[i].forceClassInst( cls, udata );
      // and see if the searched parent has something to do with them.
      udata = cls->getParentData( parent, udata );
      if( udata != 0 )
      {
         // success!
         return udata;
      }
   }
   
   // No-luck
   return 0;
}


void FlexyClass::gcMark( void* self, uint32 mark ) const
{
   static_cast<FlexyDict*>(self)->gcMark(mark);
}


void FlexyClass::enumerateProperties( void* self, PropertyEnumerator& cb ) const
{
   static_cast<FlexyDict*>(self)->enumerateProps( cb );
}

void FlexyClass::enumeratePV( void* self, PVEnumerator& cb ) const
{
   static_cast<FlexyDict*>(self)->enumeratePV( cb );
}

bool FlexyClass::hasProperty( void* self, const String& prop ) const
{
   return static_cast<FlexyDict*>(self)->hasProperty( prop );
}


void FlexyClass::describe( void* self, String& target, int depth, int maxlen ) const
{
   String tgt;
   if( depth != 0 )
   {
      static_cast<FlexyDict*>(self)->describe( tgt, depth, maxlen );
      target.size(0);
      target += name() + "{" + tgt + "}";
   }
   else
   {
      tgt = name() + "{...}";
   }
}


void FlexyClass::op_create( VMContext* ctx, int32 pcount ) const
{
   static Collector* coll = Engine::instance()->collector();

   FlexyDict* self = new FlexyDict;
   // In case of a single parameter...
   if( pcount >= 1 )
   {
      //... it can be a dictionary or a generic class.
      Item& param = ctx->opcodeParam(0);
      Class* cls;
      void* data;
      param.forceClassInst( cls, data );

      if( cls->typeID() == FLC_CLASS_ID_DICT )
      {
         class Enum: public ItemDict::Enumerator
         {
         public:
            Enum( const FlexyClass* owner, FlexyDict* self):
               m_owner(owner),
               m_self(self),
               m_fself( owner, self)
            {
               m_fself.garbage();
            }

            virtual void operator()( const Item& key, Item& value )
            {
               if( key.isString() )
               {
                  if( key.asString()->find( ' ' ) == String::npos )
                  {
                     if( value.isFunction() )
                     {
                        Function* func = value.asFunction();
                        Item temp = m_fself;
                        temp.methodize( func );
                        m_self->insert( *key.asString(), temp );
                     }
                     else
                     {
                        m_self->insert( *key.asString(), value );
                     }
                  }
               }
            }

         private:
            const FlexyClass* m_owner;
            FlexyDict* m_self;
            Item m_fself;
         };

         Enum rator( this, self );
         ItemDict& id = *static_cast<ItemDict*>( data );
         id.enumerate( rator );
      }
      else
      {
         class Enum: public Class::PVEnumerator
         {
         public:
            Enum( const FlexyClass* owner, FlexyDict* self):
               m_owner(owner),
               m_self(self),
               m_fself( owner, self)
            {
               m_fself.garbage();
            }

            virtual void operator()( const String& data, Item& value )
            {              
               if( value.isFunction() )
               {
                  Function* func = value.asFunction();
                  Item temp = m_fself;
                  temp.methodize( func );
                  m_self->insert( data, temp );
               }
               else
               {
                  m_self->insert( data, value );
               }
            }
            
         private:
            const FlexyClass* m_owner;
            FlexyDict* m_self;
            Item m_fself;
         };

         Enum rator( this, self );
         cls->enumeratePV( data, rator );
      }

   }

   ctx->stackResult( pcount+1, FALCON_GC_STORE(coll, this, self ) );
}


void FlexyClass::op_getProperty( VMContext* ctx, void* self, const String& prop) const
{
   FlexyDict& dict = *static_cast<FlexyDict*>(self);
   Item* result = dict.find( prop );

   if( result != 0 )
   {
      ctx->topData() = *result; // should be already copied by insert
   }
   else
   {
      // fall back to the starndard system
      Class::op_getProperty( ctx, self, prop );
   }
}


void FlexyClass::op_setProperty( VMContext* ctx, void* self, const String& prop ) const
{
   FlexyDict& dict = *static_cast<FlexyDict*>(self);

   ctx->popData();
   Item& value = ctx->topData();
   if ( value.isFunction() )
   {
      Function* func = value.asFunction();
      value.setUser( this, &dict, true );
      value.methodize( func );
   }
   
   dict.insert( prop, value );
}


//=========================================================================
//
//

inline bool FlexyClass::operand( int opCount, const String& name, VMContext* ctx, void* self, bool bRaise ) const
{
   FlexyDict& dict = *static_cast<FlexyDict*>(self);
   Item* item = dict.find( name );
   if( item != 0 )
   {
      if( item->isFunction() ) 
      {
         Function* f = item->asFunction();
         Item &iself = ctx->opcodeParam(opCount-1);
         iself.methodize(f);
         ctx->call(f, opCount-1, iself );
      }
      else
      {
         // try to call the item.
         Class* cls;
         void* data;
         item->forceClassInst( cls, data );
         ctx->opcodeParam(opCount-1) = *item;
         cls->op_call( ctx, opCount - 1, data );
      }
      return true;
   }
    
   if( bRaise )
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC )
         .extra(name)
         .origin( ErrorParam::e_orig_vm)
         );
   }
   return false;
}


void FlexyClass::op_neg( VMContext* ctx, void* self ) const
{
   operand( 1, OVERRIDE_OP_NEG, ctx, self );
}

void FlexyClass::op_add( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_ADD, ctx, self );
}


void FlexyClass::op_sub( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_SUB, ctx, self );
}


void FlexyClass::op_mul( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_MUL, ctx, self );
}


void FlexyClass::op_div( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_DIV, ctx, self );
}

void FlexyClass::op_mod( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_MOD, ctx, self );
}

void FlexyClass::op_pow( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_POW, ctx, self );
}

void FlexyClass::op_shr( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_SHR, ctx, self );
}

void FlexyClass::op_shl( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_SHL, ctx, self );
}

void FlexyClass::op_aadd( VMContext* ctx, void* self) const
{
   operand( 2, OVERRIDE_OP_AADD, ctx, self );
}

void FlexyClass::op_asub( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_ASUB, ctx, self );
}

void FlexyClass::op_amul( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_AMUL, ctx, self );
}

void FlexyClass::op_adiv( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_ADIV, ctx, self );
}

void FlexyClass::op_amod( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_AMOD, ctx, self );
}

void FlexyClass::op_apow( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_APOW, ctx, self );
}

void FlexyClass::op_ashr( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_ASHR, ctx, self );
}

void FlexyClass::op_ashl( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_ASHL, ctx, self );
}

void FlexyClass::op_inc( VMContext* ctx, void* self ) const
{
   operand( 1, OVERRIDE_OP_INC, ctx, self );
}

void FlexyClass::op_dec( VMContext* ctx, void* self) const
{
   operand( 1, OVERRIDE_OP_DEC, ctx, self );
}

void FlexyClass::op_incpost( VMContext* ctx, void* self ) const
{
   operand( 1, OVERRIDE_OP_INCPOST, ctx, self );
}

void FlexyClass::op_decpost( VMContext* ctx, void* self ) const
{
   operand( 1, OVERRIDE_OP_DECPOST, ctx, self );
}

void FlexyClass::op_getIndex( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_GETINDEX, ctx, self );
}

void FlexyClass::op_setIndex( VMContext* ctx, void* self ) const
{
   operand( 3, OVERRIDE_OP_SETINDEX, ctx, self );
}


void FlexyClass::op_compare( VMContext* ctx, void* self ) const
{
   if ( ! operand( 2, OVERRIDE_OP_COMPARE, ctx, self, false ) )
   {
      Class::op_isTrue(ctx, self);
   }
}

void FlexyClass::op_isTrue( VMContext* ctx, void* self ) const
{
   if ( ! operand( 1, OVERRIDE_OP_ISTRUE, ctx, self, false ) )
   {
      Class::op_isTrue(ctx, self);
   }
}

void FlexyClass::op_in( VMContext* ctx, void* self ) const
{
   operand( 2, OVERRIDE_OP_IN, ctx, self );
}

void FlexyClass::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   operand( paramCount+1, OVERRIDE_OP_CALL, ctx, self );   
}

void FlexyClass::op_toString( VMContext* ctx, void* self ) const
{
   if ( ! operand( 1, OVERRIDE_OP_TOSTRING, ctx, self, false ) )
   {
      Class::op_toString(ctx, self);
   }
}

void FlexyClass::op_iter( VMContext* ctx, void* self ) const
{
   // copy the topdata
   ctx->addSpace(1);
   ctx->opcodeParam(0) = ctx->opcodeParam(1);
   operand( 1, OVERRIDE_OP_ITER, ctx, self );
}


void FlexyClass::op_next( VMContext* ctx, void* self ) const
{
   // copy the 2 items at the top 
   ctx->addSpace(2);
   ctx->opcodeParam(0) = ctx->opcodeParam(2);
   ctx->opcodeParam(1) = ctx->opcodeParam(3);
   operand( 2, OVERRIDE_OP_NEXT, ctx, self );
}

   
}

/* end of flexyclass.cpp */
