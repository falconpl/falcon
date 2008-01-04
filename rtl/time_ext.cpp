/*
   FALCON - The Falcon Programming Language.
   FILE: time_ext.cpp
   $Id: time_ext.cpp,v 1.10 2007/08/11 00:11:57 jonnymind Exp $

   Date and time support for RTL
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven nov 12 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Date and time support for RTL.
*/

#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/sys.h>
#include <falcon/symbol.h>
#include <math.h>
#include <falcon/cobject.h>
#include <falcon/fassert.h>

#include <falcon/timestamp.h>
#include <falcon/time_sys.h>

namespace Falcon { namespace Ext {

/** Date constructor:
   TimeStamp_init( ) -- creates an empty date
   TimeStamp_init( int ) -- use a date in long format
   TimeStamp_init( date_object ) -- copies an existing date
*/


bool TimeStamp_copy( const CoreObject *origin, CoreObject *dateObj )
{
   Item data;
   origin->getProperty( "year", data );
   dateObj->setProperty( "year", data );
   origin->getProperty( "month", data );
   dateObj->setProperty( "month", data );
   origin->getProperty( "day", data );
   dateObj->setProperty( "day", data );
   origin->getProperty( "hour", data );
   dateObj->setProperty( "hour", data );
   origin->getProperty( "minute", data );
   dateObj->setProperty( "minute", data );
   origin->getProperty( "second", data );
   dateObj->setProperty( "second", data );
   origin->getProperty( "msec", data );
   dateObj->setProperty( "msec", data );
   return true;
}



FALCON_FUNC  TimeStamp_init ( ::Falcon::VMachine *vm )
{
   Item *date = vm->param(0);
   CoreObject *self = vm->self().asObject();

   if ( date != 0 ) {
      if ( date->isInteger() ) {
         self->setUserData( new TimeStamp( date->asInteger() ) );
      }
      else if ( date->isObject() ) {
         CoreObject *other = date->asObject();
         if( !other->derivedFrom( "TimeStamp" ) ) {
            vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
               extra( "Parameter is not a TimeStamp" ) ) );
         }
         self->setUserData( new TimeStamp );

         TimeStamp_copy( other, self );
      }
      else {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
               extra( "TimeStamp class init requires an integer or TimeStamp parameter" ) ) );
      }
   }
   else
      self->setUserData( new TimeStamp );
}

FALCON_FUNC  TimeStamp_currentTime ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   TimeStamp *ts = (TimeStamp *) self->getUserData();
   Falcon::Sys::Time::currentTime( *ts );
}


FALCON_FUNC  TimeStamp_toString ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   TimeStamp *ts = (TimeStamp *) self->getUserData();
   Item *format = vm->param( 0 );

   String *str = new GarbageString( vm );
   if( format != 0 )
   {
      if( ! format->isString() )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
            extra( "[S]" ) ) );
         return;
      }

      if( !  ts->toString( *str, *format->asString() ) )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
            extra( "Invalid TimeStamp format" ) ) );
         return;
      }
   }
   else {
      ts->toString( *str );
   }
   vm->retval( str );
}

static void internal_add_dist( ::Falcon::VMachine *vm, int mode )
{
   CoreObject *self = vm->self().asObject();
   TimeStamp *ts1, *ts2;
   ts1 = (TimeStamp *) self->getUserData();
   Item *date = vm->param( 0 );

   if ( date->isObject() )
   {
      CoreObject *other = date->asObject();
      if( !other->derivedFrom( "TimeStamp" ) )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
               extra( "not a TimeStamp" ) ) );
         return;
      }


      ts2 = (TimeStamp *) date->asObject()->getUserData();
      if ( mode == 0 )
         ts1->add( *ts2 );
      else
         ts1->distance( *ts2 );
   }
   else {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
            extra( "Not a TimeStamp" ) ) );
   }
}

FALCON_FUNC  TimeStamp_add ( ::Falcon::VMachine *vm )
{
   internal_add_dist( vm, 0 );
}

FALCON_FUNC  TimeStamp_distance ( ::Falcon::VMachine *vm )
{
   internal_add_dist( vm, 1 );
}

FALCON_FUNC  TimeStamp_isValid ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   TimeStamp *ts = (TimeStamp *) self->getUserData();
   vm->retval( ts->isValid() );
}

FALCON_FUNC  TimeStamp_isLeapYear ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   TimeStamp *ts = (TimeStamp *) self->getUserData();
   vm->retval( ts->isLeapYear() );
}

FALCON_FUNC  TimeStamp_dayOfWeek ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   TimeStamp *ts = (TimeStamp *) self->getUserData();
   vm->retval( ts->dayOfWeek() );
}

FALCON_FUNC  TimeStamp_dayOfYear ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   TimeStamp *ts = (TimeStamp *) self->getUserData();
   vm->retval( ts->dayOfYear() );
}

FALCON_FUNC  TimeStamp_toLongFormat ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   TimeStamp *ts = (TimeStamp *) self->getUserData();
   vm->retval( ts->toLongFormat() );
}


FALCON_FUNC  TimeStamp_fromLongFormat ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   TimeStamp *ts = (TimeStamp *) self->getUserData();

   Item *data = vm->param( 0 );

   if ( ! data->isInteger() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
         extra( "Only integer parameter allowed" ) ) );
   }

   ts->fromLongFormat( data->asInteger() );
}

FALCON_FUNC  TimeStamp_compare ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   TimeStamp *ts1, *ts2;
   ts1 = (TimeStamp *) self->getUserData();
   Item *date = vm->param( 0 );

   if ( date->isObject() )
   {
      CoreObject *other = date->asObject();
      if( other->derivedFrom( "TimeStamp" ) )
      {
         ts2 = (TimeStamp *) date->asObject()->getUserData();
         vm->retval( ts1->compare( *ts2 ) );
      }
      else {
         vm->retval( vm->self().compare( *date ) );
      }
   }
   else {
      vm->retval( vm->self().compare( *date ) );
   }
}


/** Factory function for a timestamp defined now.  */
FALCON_FUNC  CurrentTime ( ::Falcon::VMachine *vm )
{
   // create the timestamp
   Item *ts_class = vm->findWKI( "TimeStamp" );
   //if we wrote the std module, can't be zero.
   fassert( ts_class != 0 );
   CoreObject *self = ts_class->asClass()->createInstance();
   TimeStamp *ts = new TimeStamp;

   Falcon::Sys::Time::currentTime( *ts );
   self->setUserData( ts );
   vm->retval( self );
}

}}

/* end of time_ext.cpp */
