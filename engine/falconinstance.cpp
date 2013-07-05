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
#include <falcon/psteps/exprinherit.h>
#include <falcon/stderrors.h>


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

bool FalconInstance::getMember( const String& name, Item& target ) const
{   
   const FalconClass::Property* prop = m_origin->getProperty( name );
   if( prop == 0 )
   {
      return false;
   }

   switch( prop->m_type )
   {
      case FalconClass::Property::t_prop:
         target.copyInterlocked(m_data[ prop->m_value.id ]);
         if( target.isFunction() ) {
            Function* func = target.asFunction();
            target.setUser( m_origin, const_cast<FalconInstance*>(this) );
            target.methodize( func );
         }
         break;

      case FalconClass::Property::t_func:
         target.setUser( m_origin, const_cast<FalconInstance*>(this) );
         target.methodize( prop->m_value.func );
         break;

      case FalconClass::Property::t_inh:
         target.setUser( prop->m_value.inh->base(), const_cast<FalconInstance*>(this) );
         break;

      case FalconClass::Property::t_state:
         //TODO
         break;
   }

   return true;
}

void FalconInstance::setProperty( const String& name, const Item& value )
{
   const FalconClass::Property* prop = m_origin->getProperty( name );
   if( prop == 0 )
   {
      throw new AccessError( ErrorParam( e_prop_acc, __LINE__, __FILE__ ).extra( name ) );
   }

   if( prop->m_type != FalconClass::Property::t_prop )
   {
      throw new AccessTypeError( ErrorParam( e_prop_ro, __LINE__, __FILE__ ).extra( name ) );
   }

   m_data[ prop->m_value.id ].copyInterlocked( value );
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
      // also, back-mark our class.
      // possibly, it's our class that's marking ourselves,
      // but we don't care. This will be a no-op in that case.
      const_cast<FalconClass*>(m_origin)->gcMark( mark );
   }
}

}

/* end of falconinstance.cpp */
