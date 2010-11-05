/*
   FALCON - The Falcon Programming Language.
   FILE: coreobject.cpp

   Core object implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 27 Jan 2009 19:46:05 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Core object implementation.
*/

#include <falcon/vm.h>
#include <falcon/item.h>
#include <falcon/coreobject.h>
#include <falcon/symbol.h>
#include <falcon/cclass.h>
#include <falcon/stream.h>
#include <falcon/falcondata.h>

namespace Falcon
{

CoreObject::CoreObject( const CoreClass *parent ):
   Garbageable(),
   m_user_data( 0 ),
   m_bIsFalconData( false ),
   m_bIsSequence( false ),
   m_generatedBy( parent ),
   m_state( 0 )
{
}

CoreObject::CoreObject( const CoreObject &other ):
   Garbageable( other ),
   m_user_data( 0 ),
   m_bIsFalconData( other.m_bIsFalconData ),
   m_bIsSequence( other.m_bIsFalconData ),
   m_generatedBy( other.m_generatedBy ),
   m_state( 0 )
{
   if ( m_bIsFalconData )
   {
      fassert( other.m_user_data != 0 );
      m_user_data = other.getFalconData()->clone();
   }
   else {
      fassert( other.m_user_data == 0 );
   }
}


CoreObject::~CoreObject()
{
   delete m_state;

   if ( m_bIsFalconData )
      delete static_cast<FalconData *>( m_user_data );
}


void CoreObject::gcMark( uint32 gen )
{
   // our class
   const_cast<CoreClass*>(m_generatedBy)->gcMark( gen );

   if( gen != mark() )
   {
      // mark ourseleves
      mark( gen );

      // and possibly our inner falcon data
      if ( m_bIsFalconData )
      {
         fassert( m_user_data != 0 );
         static_cast<FalconData* >(m_user_data)->gcMark( gen );
      }
   }
}


bool CoreObject::serialize( Stream *stream, bool bLive ) const
{
   if( bLive )
   {
      void* data = 0;
      if( m_bIsFalconData )
      {
         data = getFalconData()->clone();
      }

      if ( data == 0 )
      {
         data = m_user_data;
      }

      stream->write( (byte *) &data, sizeof( m_user_data ) );
      return true;
   }
   return false;
}


bool CoreObject::hasProperty( const String &key ) const
{
   fassert( m_generatedBy != 0 );

   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();
   return pt.findKey( key, pos );
}


bool CoreObject::defaultProperty( const String &key, Item &prop ) const
{
   fassert( m_generatedBy != 0 );

   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();
   if ( pt.findKey( key, pos ) )
   {
      prop = *pt.getValue(pos);
      return true;
   }

   return false;
}


void CoreObject::readOnlyError( const String &key ) const
{
   fassert( m_generatedBy != 0 );

   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();
   throw new AccessError( ErrorParam( pt.findKey( key, pos ) ? e_prop_ro : e_prop_acc, __LINE__ )
         .extra( key ) );
}

bool CoreObject::deserialize( Stream *stream, bool bLive )
{
   if( bLive )
   {
      if ( stream->read( (byte *) &m_user_data, sizeof( m_user_data ) ) != sizeof( m_user_data ) )
         return false;
      return true;
   }

   return false;
}



bool CoreObject::derivedFrom( const String &className ) const
{
   return m_generatedBy->derivedFrom( className );
}


bool CoreObject::getMethodDefault( const String &name, Item &mth ) const
{
   const Falcon::Item* pmth = generator()->properties().getValue( name );

   if ( pmth != 0 && pmth->isFunction() )
   {
      // yes, a valid method
      mth = *pmth;

      mth.methodize( SafeItem( const_cast<CoreObject*>(this) ) );
      return true;
   }
   return false;
}


bool CoreObject::apply( const ItemDict& dict, bool bRaiseOnError )
{
   Iterator iter( const_cast<ItemDict*>(&dict) );
   bool bRes = true;

   while( iter.hasCurrent() )
   {
      const Item& key = iter.getCurrentKey();
      if ( key.isString() )
      {
         if ( ! setProperty( *key.asString(), iter.getCurrent() ) )
         {
            if( bRaiseOnError )
            {
               throw new AccessError( ErrorParam( e_prop_acc, __LINE__ )
                     .origin( e_orig_runtime )
                     .extra( *key.asString() ) );
            }
            else
               bRes = false;
         }
      }

      iter.next();
   }

   return bRes;
}


bool CoreObject::retrieve( ItemDict& dict, bool bRaiseOnError, bool bFillDict, bool bIgnoreMethods ) const
{
   if ( bFillDict )
   {
      const PropertyTable& names = m_generatedBy->properties();
      for ( uint32 p = 0; p < names.added(); ++p )
      {
         Item prop;
         if( getProperty( *names.getKey(p), prop ) )
         {
            if ( bIgnoreMethods && prop.canBeMethod() )
               continue;

            dict.put(  *names.getKey(p), prop );
         }
      }

      return true;
   }
   else
   {
      Iterator iter(&dict);
      bool bRes = true;

      while( iter.hasCurrent() )
      {
         const Item& key = iter.getCurrentKey();
         if ( key.isString() )
         {
            Item value;
            if ( getProperty( *key.asString(), value ) )
            {
               iter.data( &value );
            }
            else
            {
               if( bRaiseOnError )
               {
                  throw new AccessError( ErrorParam( e_prop_acc, __LINE__ )
                        .origin( e_orig_runtime )
                        .extra( *key.asString() ) );
               }
               else
                  bRes = false;
            }

         }

         iter.next();
      }

      return bRes;
   }
}

//=======================================================================
// Deep item overloading
//=======================================================================


bool CoreObject::setProperty( const String &propName, const String &value )
{
   return setProperty( propName, new CoreString( value ) );
}

void CoreObject::readIndex( const Item &pos, Item &target )
{
   Item mth;
   if ( getMethod( OVERRIDE_OP_GETINDEX, mth ) )
   {
      VMachine* vm = VMachine::getCurrent();
      if ( vm != 0 )
      {
         vm->pushParam( pos );
         vm->callItemAtomic( mth, 1 );
         target = vm->regA();
         return;
      }
   }

   throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( OVERRIDE_OP_GETINDEX ) );
}

void CoreObject::writeIndex( const Item &pos, const Item &target )
{
   Item method;
   if ( getMethod( OVERRIDE_OP_SETINDEX, method ) )
   {
      VMachine* vm = VMachine::getCurrent();
      if ( vm != 0 )
      {
         vm->pushParam( pos );
         vm->pushParam( target );
         vm->callItemAtomic( method, 2 );
         return;
      }
   }

   throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "setIndex__" ) );
}

void CoreObject::readProperty( const String &prop, Item &target )
{
   Item *p;

   if ( ! getProperty( prop, target ) )
   {
      // try to find a generic method
      VMachine* vm = VMachine::getCurrent();
      fassert( vm != 0 );
      CoreClass* cc = vm->getMetaClass( FLC_ITEM_OBJECT );
      uint32 id;
      if ( cc == 0 || ! cc->properties().findKey( prop, id ) )
      {
         throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( prop ) );
      }

      p = cc->properties().getValue( id );
   }
   else
      p = target.dereference();

   switch( p->type() ) {

      case FLC_ITEM_CLASS:
         if ( derivedFrom( p->asClass()->symbol()->name() ) )
            target.setClassMethod( this, p->asClass() );
         else
            target.setClass( p->asClass() );
      break;

      default:
        target = *p;
        target.methodize( this );
   }
}

void CoreObject::writeProperty( const String &prop, const Item &target )
{
   if ( ! setProperty( prop, target ) )
   {
      throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( prop ) );
   }
}


static bool __leave_handler( VMachine* vm )
{
   // clear enter-leave to grant a correct apply
   CoreObject* self = vm->self().asObject();
   self->setProperty("__leave", Item());
   self->setProperty("__enter", Item());

   String* state = vm->param(0)->asString();
   bool wasInState = self->hasState();
   String oldStateName;

   if( wasInState )
   {
      oldStateName = self->state();
   }
   // enter the state
   self->setState( *state, &vm->preParam(0)->asDict()->items() );

   // Does the new state have a "__enter" ?
   Item enterItem;
   if( self->getMethod("__enter", enterItem ) )
   {
      if( wasInState )
         vm->pushParam( new CoreString( oldStateName ) );
      else
         vm->pushParam( Item() );
      vm->pushParam( vm->regA() );
      // never return here
      vm->returnHandler(0);
      vm->callFrame( enterItem, 2 );
      // but respect this frame.
      return true;
   }

   // else, we're done, just remove this frame
   return false;
}


void CoreObject::setState( const String& state, VMachine* vm )
{
   ItemDict *states = m_generatedBy->states();
   Item* stateDict;
   if ( states == 0 ||
        ( stateDict = states->find( Item(const_cast<String*>(&state)) ) ) == 0
        )
   {
      throw new CodeError( ErrorParam( e_undef_state, __LINE__ )
            .origin( e_orig_runtime )
            .extra( state ) );
   }

   // do we have an active __leave property?
   Item leaveItem;
   if( getMethod("__leave", leaveItem ) )
   {
      CoreString* csState = new CoreString( state );
      vm->pushParam(*stateDict); // pre-param 0
      vm->pushParam( csState );
      vm->callFrame( leaveItem, 1, &__leave_handler );
      return;
   }

   // no leave? -- apply and see if we have an enter.
   setProperty("__enter", Item());

   fassert( stateDict->isDict() );
   bool hasOldState;
   String sOldState;

   if ( m_state == 0 )
   {
      hasOldState = false;
      m_state = new String( state );
   }
   else
   {
      hasOldState = true;
      sOldState = *m_state;
      m_state->bufferize( state );
   }

   // shouldn't raise if all is ok
   apply( stateDict->asDict()->items(), true );

   Item enterItem;
   if( getMethod("__enter", enterItem ) )
   {
      if( hasOldState )
         vm->pushParam( sOldState );
      else
         vm->pushParam( Item() );

      vm->pushParam( Item() );
      vm->callFrame( enterItem, 2 );
   }
}

void CoreObject::setState( const String& state, ItemDict* stateDict )
{
   if ( m_state == 0 )
      m_state = new String( state );
   else
      m_state->bufferize( state );

   apply( *stateDict, true );
}

void CoreObject::setUserData( FalconData* fdata )
{
   fdata->gcMark( mark() );
   m_bIsSequence = false;
   m_bIsFalconData = true;
   m_user_data = fdata;
}



void CoreObject::setUserData( Sequence* sdata )
{
   sdata->gcMark( mark() );
   m_bIsSequence = true;
   m_bIsFalconData = true;
   m_user_data = sdata;
}

}

/* end of coreobject.cpp */
