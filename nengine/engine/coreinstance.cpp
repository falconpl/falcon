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

#include <falcon/coreinstance.h>
#include <falcon/itemid.h>
#include <falcon/falconclass.h>
#include <falcon/falconinstance.h>
#include <falcon/optoken.h>
#include <falcon/vm.h>
#include <falcon/operanderror.h>

namespace Falcon
{

CoreInstance::CoreInstance():
   Class("Object", FLC_CLASS_ID_OBJECT )
{
}


CoreInstance::~CoreInstance()
{
}


//=========================================
// Instance management

void* CoreInstance::create( void* creationParams ) const
{
   FalconClass* cls = static_cast<cpars*>(creationParams)->flc;
   return cls->createInstance();
}


void CoreInstance::dispose( void* self ) const
{
   delete static_cast<FalconInstance*>(self);
}


void* CoreInstance::clone( void* source ) const
{
   return static_cast<FalconInstance*>(source)->clone();
}


void CoreInstance::serialize( DataWriter* stream, void* self ) const
{
   static_cast<FalconInstance*>(self)->serialize(stream);
}


void* CoreInstance::deserialize( DataReader* stream ) const
{
   // TODO
   FalconInstance* fi = new FalconInstance;
   try
   {
      fi->deserialize(stream);
   }
   catch( ... )
   {
      delete fi;
      throw;
   }
   return fi;
}



//=========================================================
// Class management
//

void CoreInstance::gcMark( void* self, uint32 mark ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   inst->gcMark( mark );
}


void CoreInstance::enumerateProperties( void* self, PropertyEnumerator& cb ) const
{
   // the instance has the properties declared in its class.
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   const FalconClass* orig = inst->origin();
   orig->enumerateMembers( cb );
}


bool CoreInstance::hasProperty( void* self, const String& prop ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   const FalconClass* orig = inst->origin();
   return orig->hasMember( prop );
}


void CoreInstance::describe( void* instance, String& target, int depth, int maxlen ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(instance);
   const FalconClass* orig = inst->origin();


   class Descriptor: public FalconClass::PropertyEnumerator
   {
   public:
      Descriptor( FalconInstance* inst, String& forming, int d, int l ):
         m_inst(inst),
         m_target( forming ),
         m_depth( d ),
         m_maxlen( l )
      {}
         
      virtual bool operator()( const String& name, bool bLast )
      {
         Item theItem;
         m_inst->getMember( name, theItem );
         String temp;
         theItem.describe( temp, m_depth-1, m_maxlen );
         m_target += name + "=" + temp;
         if( ! bLast )
         {
            m_target += ", ";
         }
         return true;
      }

   private:
      FalconInstance* m_inst;
      String& m_target;
      int m_depth;
      int m_maxlen;
   };

   Descriptor rator( inst, target, depth, maxlen );

   target = "Instance of " + orig->name() +"{" ;
   orig->enumerateProperties( rator );
   target += "}";
}


//=========================================================
// Operators.
//

inline void override_unary( VMachine *vm, void* self, const String& op )
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   Item item;
   if( inst->origin()->getMember( op, item ) && item.isFunction() )
   {
      vm->call (item.asFunction(), 0, vm->currentContext()->topData(), true );
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(op) );
   }
}

inline void override_binary( VMachine *vm, void* self, const String& op )
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   Item item;
   if( inst->origin()->getMember( op, item ) && item.isFunction() )
   {
      Item* first, *second;
      OpToken token( vm, first, second );
      Item i_first = *first;
      vm->currentContext()->pushData(*second);
      // use the instance we know, as first can be moved away.s
      vm->call ( item.asFunction(), 1, i_first, true );
      token.abandon();
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(op) );
   }
}


void CoreInstance::op_neg( VMachine *vm, void* self ) const
{
   override_unary( vm, self, OVERRIDE_OP_NEG );
}


void CoreInstance::op_add( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_ADD );
}


void CoreInstance::op_sub( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_SUB );
}


void CoreInstance::op_mul( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_MUL );
}


void CoreInstance::op_div( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_DIV );
}

void CoreInstance::op_mod( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_MOD );
}


void CoreInstance::op_pow( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_POW );
}


void CoreInstance::op_aadd( VMachine *vm, void* self) const
{
   override_binary( vm, self, OVERRIDE_OP_AADD );
}


void CoreInstance::op_asub( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_ASUB );
}


void CoreInstance::op_amul( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_AMUL );
}


void CoreInstance::op_adiv( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_DIV );
}


void CoreInstance::op_amod( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_AMOD );
}


void CoreInstance::op_apow( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_APOW );
}


void CoreInstance::op_inc(VMachine *vm, void* self ) const
{
   override_unary( vm, self, OVERRIDE_OP_INC );
}


void CoreInstance::op_dec(VMachine *vm, void* self) const
{
   override_unary( vm, self, OVERRIDE_OP_DEC );
}


void CoreInstance::op_incpost(VMachine *vm, void* self ) const
{
   override_unary( vm, self, OVERRIDE_OP_INCPOST );
}


void CoreInstance::op_decpost(VMachine *vm, void* self ) const
{
   override_unary( vm, self, OVERRIDE_OP_DECPOST );
}


void CoreInstance::op_getIndex(VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_GETINDEX );
}


void CoreInstance::op_setIndex(VMachine *vm, void* self ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   Item item;
   if( inst->origin()->getMember( OVERRIDE_OP_SETINDEX, item ) && item.isFunction() )
   {
      Item* first, *second, *third;
      OpToken token( vm, first, second, third );
      Item i_first = *first;
      Item i_third = *third;
      vm->currentContext()->pushData(*second);
      vm->currentContext()->pushData( i_third);

      // use the instance we know, as first can be moved away.s
      vm->call( item.asFunction(), 2, i_first, true );
      token.abandon();
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(OVERRIDE_OP_SETINDEX) );
   }
}


void CoreInstance::op_getProperty( VMachine *vm, void* self, const String& prop) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   Item item;
   if( inst->origin()->getMember( OVERRIDE_OP_GETPROP, item ) )
   {
      if( ! item.isFunction() )
      {
         throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(OVERRIDE_OP_GETPROP) );
      }
      
      Item* first;
      OpToken token( vm, first );
      Item i_first = *first;
      // here we push the string as UserData, but...
      vm->currentContext()->pushData( prop ); // for sure, it's held elsewhere

      // use the instance we know, as first can be moved away.
      vm->call( item.asFunction(), 1, i_first, true );
      token.abandon();
   }
   else
   {
      inst->getMember( prop, vm->currentContext()->topData() );
   }
}


void CoreInstance::op_setProperty( VMachine *vm, void* self, const String& prop ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   Item item;
   if( inst->origin()->getMember( OVERRIDE_OP_SETPROP, item ) )
   {
      if( ! item.isFunction() )
      {
         throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(OVERRIDE_OP_SETPROP) );
      }

      Item* first, *second;
      OpToken token( vm, first, second );
      Item i_first = *first;
      Item i_second = *second;

      vm->currentContext()->pushData( prop );
      vm->currentContext()->pushData( i_second );

      // use the instance we know, as first can be moved away.
      vm->call( item.asFunction(), 2, i_first, true );
      token.abandon();
   }
   else
   {
      inst->setProperty( prop, vm->currentContext()->topData() );
   }
}


void CoreInstance::op_compare( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_COMPARE );
}


void CoreInstance::op_isTrue( VMachine *vm, void* self ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   Item item;
   if( inst->origin()->getMember( OVERRIDE_OP_ISTRUE, item ) )
   {
      if( ! item.isFunction() )
      {
         throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(OVERRIDE_OP_ISTRUE) );
      }

      Item* first;
      OpToken token( vm, first );

      // use the instance we know, as first can be moved away.
      vm->call( item.asFunction(), 0, *first, true );
      token.abandon();
   }
   else
   {
      // instances are always true.
      vm->currentContext()->topData().setBoolean(true);
   }
}


void CoreInstance::op_in( VMachine *vm, void* self ) const
{
   override_binary( vm, self, OVERRIDE_OP_IN );
}


void CoreInstance::op_provides( VMachine *vm, void* self, const String& property ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   vm->currentContext()->topData().setBoolean( inst->origin()->hasMember(property) );
}


void CoreInstance::op_call( VMachine *vm, int32 paramCount, void* self ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   Item item;
   if( inst->origin()->getMember( OVERRIDE_OP_CALL, item ) || ! item.isFunction() )
   {
      // self/inst is safely elsewhere (i.e. in the stack), we can use as UserData
      vm->call( item.asFunction(), paramCount, Item( const_cast<CoreInstance*>(this), inst ) );
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(OVERRIDE_OP_CALL) );
   }
}


void CoreInstance::op_toString( VMachine *vm, void* self ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   Item item;
   if( inst->origin()->getMember( OVERRIDE_OP_CALL, item ) || ! item.isFunction() )
   {
      // self/inst is safely elsewhere (i.e. in the stack), we can use as UserData
      vm->call( item.asFunction(), 0, Item( const_cast<CoreInstance*>(this), inst ) );
   }
   else
   {
      String* str = new String("Instance of ");
      str->append( inst->origin()->name() );
      vm->currentContext()->topData() = str;
   }
}

}

/* end of coreinstance.cpp */
