/*
   FALCON - The Falcon Programming Language.
   FILE: cmdlineparser.cpp

   The command line parser class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-11-30
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
   The command line parser class
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/carray.h>
#include <falcon/vm.h>
#include <falcon/stream.h>
#include "falcon_rtl_ext.h"
#include "rtl_messages.h"

namespace Falcon{
namespace Ext {

FALCON_FUNC  CmdlineParser_parse( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_params = vm->param( 0 );

   if ( i_params == 0 )
   {
      // get the parameters from the VM args object
      i_params = vm->findGlobalItem( "args" );
      if ( i_params == 0 || ! i_params->isArray() ) {
         vm->raiseRTError( new CodeError( ErrorParam( e_undef_sym ).extra( "args" ).hard() ) );
         return;
      }
   }
   else if ( ! i_params->isArray() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "( A )" ) ) );
      return;
   }

   CoreArray *args = i_params->asArray();

   // zero request.
   self->setProperty( "_request", (int64) 0 );
   self->setProperty( "lastParsed", (int64) 0 );

   // status.
   typedef enum {
      t_none,
      t_waitingValue,
      t_allFree
   } t_states;

   t_states state = t_none ;
   String currentOption;
   Item i_method;
   Item i_passMM;
   self->getProperty( "passMinusMinus", i_passMM );
   bool passMM = i_passMM.isTrue();
   Item _request;
   String subParam;
   uint32 i;

   for ( i = 0; i < args->length(); i++ )
   {
      Item &i_opt = args->at( i );
      if ( !i_opt.isString() )
      {
         vm->raiseRTError(
            new ParamError( ErrorParam( e_param_type ).
                  extra( getMessage( msg::rtl_cmdp_0 ) )
               )
            );

         return;
      }

      String &opt = *i_opt.asString();
       // if we were expecting a value, we MUST consider ANYTHING as it was a value.
      if ( state == t_waitingValue )
      {
         self->getProperty( "onValue", i_method );
         if ( i_method.methodize( self ) )
         {
            vm->pushParameter( &currentOption );
            vm->pushParameter( i_opt );
            vm->callItemAtomic( i_method, 2 );
            if( vm->hadEvent() )
               return;

            vm->resetEvent();
            state = t_none;
         }
         else
         {
            vm->retval( false );
            self->setProperty( "lastParsed", (int64) i );
            return;
         }
      }
      else if( opt.length() == 0 || (opt.getCharAt( 0 ) != '-' || opt.length() == 1) || state == t_allFree )
      {

         self->getProperty( "onFree", i_method );
         if ( i_method.methodize( self ) )
         {
            vm->pushParameter( i_opt );
            vm->callItemAtomic( i_method, 1 );
            if( vm->hadEvent() )
               return;
            vm->resetEvent();

         }
         else
         {
            vm->retval( false );
            self->setProperty( "lastParsed", (int64) i );
            return;
         }
      }
      else if ( opt == "--" && ! passMM )
      {
         state = t_allFree;
         continue; // to skip return value.
      }
      else {
         // we have at least one '-', and length > 1
         if ( opt.getCharAt( 1 ) == (uint32) '-' )
         {
            self->getProperty( "onOption", i_method );

            if ( i_method.methodize( self ) )
            {
               if ( passMM && opt.size() == 2 )
                  vm->pushParameter( i_opt );
               else {
                  //Minimal optimization; reuse the same string and memory
                  subParam = opt.subString( 2 );
                  vm->pushParameter( &subParam );
               }

               vm->callItemAtomic( i_method, 1 );
               if( vm->hadEvent() )
                  return;
               self->getProperty( "_request", _request );
               // value requested?
               if ( _request.asInteger() == 1 ) {
                  currentOption = subParam;
               }
            }
            else
            {
               vm->retval( false );
               self->setProperty( "lastParsed", (int64) i );
               return;
            }
         }
         else {
            // we have a switch set.
            for( uint32 chNum = 1; chNum < opt.length(); chNum++ )
            {
               //Minimal optimization; reuse the same string and memory

               subParam.size( 0 );
               subParam.append( opt.getCharAt( chNum ) );

               if ( chNum < opt.length() -1 && opt.getCharAt( chNum +1 ) == (uint32) '-' )
               {
                  // switch turnoff.
                  self->getProperty( "onSwitchOff", i_method );
                  if ( i_method.methodize( self ) )
                  {
                     vm->pushParameter( &subParam );
                     vm->callItemAtomic( i_method, 1 );
                     if( vm->hadEvent() )
                        return;
                 }
                  else
                  {
                     vm->retval( false );
                     self->setProperty( "lastParsed", (int64) i );
                     return;
                  }
                  chNum++;
               }
               else {
                  self->getProperty( "onOption", i_method );
                  if ( i_method.methodize( self ) )
                  {
                     vm->pushParameter( &subParam );
                     vm->callItemAtomic( i_method, 1 );
                     if( vm->hadEvent() )
                        return;
                  }
                  else
                  {
                     vm->retval( false );
                     self->setProperty( "lastParsed", (int64) i );
                     return;
                  }
               }

               self->getProperty( "_request", _request );
               // value requested?
               if ( _request.asInteger() == 1 ) {
                  currentOption = subParam;
               }
            }
         }

         self->getProperty( "_request", _request );
         // value requested?
         if ( _request.asInteger() == 1 ) {
            state = t_waitingValue;
            self->setProperty( "_request", (int64) 0 );
         }
         // or request to terminate?
         else if ( _request.asInteger() == 2 )
         {
            self->setProperty( "_request", (int64) 0 );
            vm->retval( true );
            self->setProperty( "lastParsed", (int64) i );
            return;
         }
      }
   }

   self->setProperty( "lastParsed", (int64) i );
   vm->resetEvent();
   vm->retval( true );
}

FALCON_FUNC  CmdlineParser_expectValue( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   self->setProperty( "_request", (int64) 1 );
}

FALCON_FUNC  CmdlineParser_terminate( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   self->setProperty( "_request", (int64) 2 );
}

FALCON_FUNC  CmdlineParser_usage( ::Falcon::VMachine *vm )
{
   vm->stdErr()->writeString( "The stub for \"CmdlineParser.usage()\" has been called.\n" );
   vm->stdErr()->writeString( "This class should be derived and the method usage() overloaded.\n" );
}

}}

/* end of cmdlineparser.cpp */
