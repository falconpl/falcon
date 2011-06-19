/*
   FALCON - The Falcon Programming Language.
   FILE: falconinstance.cpp

   Instance of classes declared in falcon scripts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jun 2011 14:35:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falconinstance.h>
#include <falcon/falconclass.h>
#include <falcon/accesserror.h>
#include <falcon/accesstypeerror.h>

namespace Falcon
{

FalconInstance::FalconInstance():
   m_origin(0),
   m_mark(0)
{
}

FalconInstance::FalconInstance( const FalconClass* origin ):
   m_origin(origin),
   m_mark(0)
{
}


FalconInstance::FalconInstance( const FalconInstance& other ):
   m_data(other.m_data),
   m_origin(other.m_origin),
   m_mark(other.m_mark)
{}

FalconInstance::~FalconInstance()
{
}

void FalconInstance::getMember( const String& name, Item& target ) const
{
   static Class* cinst = Engine::instance()->instanceClass();
   static Collector* coll = Engine::instance()->collector();
   
   const FalconClass::Property* prop = m_origin->getMember( name );
   if( prop == 0 )
   {
      throw new AccessError( ErrorParam( e_prop_acc, __LINE__, __FILE__ ).extra( name ) );
   }

   switch( prop->m_type )
   {
      case FalconClass::Property::t_prop:
      case FalconClass::Property::t_expr:
         target = m_data[ prop->m_value.id ];
         break;

      case FalconClass::Property::t_func:
         target.setDeep( coll->store( cinst, (void*) this ) );
         target.methodize( prop->m_value.func );
         break;

      case FalconClass::Property::t_inh:
         target.setDeep( coll->store( cinst, (void*) this ) );
         //TODO
         //target.methodize( prop.m_value.inh. somethin );
         break;

      case FalconClass::Property::t_state:
         //TODO
         break;
   }
}

void FalconInstance::setProperty( const String& name, const Item& value )
{
   const FalconClass::Property* prop = m_origin->getMember( name );
   if( prop == 0 )
   {
      throw new AccessError( ErrorParam( e_prop_acc, __LINE__, __FILE__ ).extra( name ) );
   }

   if( prop->m_type != FalconClass::Property::t_prop )
   {
      throw new AccessTypeError( ErrorParam( e_prop_ro, __LINE__, __FILE__ ).extra( name ) );
   }

   m_data[ prop->m_value.id ].assign( value );
}

void FalconInstance::serialize( DataWriter* ) const
{
}

void FalconInstance::deserialize( DataReader* )
{
}


void FalconInstance::gcMark( uint32 mark )
{
   if( mark != m_mark )
   {
      m_mark = mark;
      m_data.gcMark( mark );
      m_origin->gcMark( mark );
   }
}

}

/* end of falconinstance.cpp */
