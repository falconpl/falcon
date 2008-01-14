/*
   FALCON - The Falcon Programming Language
   FILE: compiler_ext.cpp
   $Id: compiler_ext.cpp,v 1.12 2007/08/11 19:02:32 jonnymind Exp $

   Compiler module main file - extension implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab lug 21 2007
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
   Compiler module main file - extension implementation.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/stringstream.h>

#include "compiler_ext.h"
#include "compiler_mod.h"

namespace Falcon {

namespace Ext {

FALCON_FUNC Compiler_init( ::Falcon::VMachine *vm )
{
   Item *i_path = vm->param( 0 );

   CompilerIface *iface;

   if( i_path != 0 )
   {
      if( ! i_path->isString() )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "[S]" ) ) );
         return;
      }

      iface = new CompilerIface( vm->self().asObject(), *i_path->asString() );
   }
   else
      iface = new CompilerIface( vm->self().asObject() );

   // set our VM as the error handler for this loader.
   iface->loader().errorHandler( vm );

   vm->self().asObject()->setUserData( iface );
}


void internal_link( ::Falcon::VMachine *vm, Module *mod, CompilerIface *iface )
{

   Runtime rt( &iface->loader(), vm );

   // let's try to link
   if ( ! rt.addModule( mod ) || ! vm->link( &rt ) )
   {
      // VM should have raised the errors.
      mod->decref();
      vm->retnil();
      return;
   }

   // ok, the module is up and running.
   // wrap it
   Item *mod_class = vm->findWKI( "Module" );
   fassert( mod_class != 0 );
   CoreObject *co = mod_class->asClass()->createInstance();
   // we know the module IS in the VM.
   co->setUserData( new ModuleCarrier( vm->findModule( mod->name() ) ) );

   co->setProperty( "name", mod->name() );
   co->setProperty( "path", mod->path() );

   // return the object
   vm->retval( co );

   // we can drop our reference to the module
   mod->decref();
}


FALCON_FUNC Compiler_compile( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );
   // The parameter may be  a string or a stream
   Item *i_data = vm->param( 1 );

   if( i_name == 0 || ! i_name->isString() ||
      i_data == 0 || (! i_data->isString() && ! i_data->isObject()) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, S|O" ) ) );
      return;
   }

   Stream *input;
   String name;
   bool bDelete;

   // now, if data is an object it must be a stream.
   if( i_data->isObject() )
   {
      CoreObject *data = i_data->asObject();
      if ( ! data->derivedFrom( "Stream" ) )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "Object must be a stream" ) ) );
         return;
      }

      // ok, get the stream
      input = (Stream *) data->getUserData();
      name = "unknown_module";
      bDelete = false;
   }
   else {
      // if it's a string, we have to create a stream
      name = *i_data->asString();
      input = new StringStream( name );
      bDelete = true;
   }

   CoreObject *self = vm->self().asObject();
   CompilerIface *iface = static_cast<CompilerIface *>( self->getUserData() );

   Module *mod = iface->loader().loadSource( input, name );

   // if mod is zero, do nothing: vm has already raised the error.
   if ( mod != 0 )
   {
      mod->name( *i_name->asString() );
      internal_link( vm, mod, iface );
   }

   if( bDelete )
      delete input;
}

FALCON_FUNC Compiler_loadByName( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );

   if( i_name == 0 || ! i_name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   CompilerIface *iface = static_cast<CompilerIface *>( self->getUserData() );

   Module *mod = iface->loader().loadName( *i_name->asString() );

   // if mod is zero, do nothing: vm has already raised the error.
   if ( mod != 0 )
      internal_link( vm, mod, iface );
}

FALCON_FUNC Compiler_loadModule( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );

   if( i_name == 0 || ! i_name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   CompilerIface *iface = static_cast<CompilerIface *>( self->getUserData() );

   Module *mod = iface->loader().loadFile( *i_name->asString() );

   // if mod is zero, do nothing: vm has already raised the error.
   if ( mod != 0 )
      internal_link( vm, mod, iface );
}

FALCON_FUNC Compiler_setDirective( ::Falcon::VMachine *vm )
{
   Item *i_directive = vm->param( 0 );
   Item *i_value = vm->param( 1 );

   if( i_directive == 0 || ! i_directive->isString() ||
       i_value == 0 || ( ! i_value->isString() && ! i_value->isOrdinal() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S,S|N" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   CompilerIface *iface = static_cast<CompilerIface *>( self->getUserData() );
   if ( i_value->isString() )
      iface->loader().compiler().setDirective( *i_directive->asString(), *i_value->asString() );
   else
      iface->loader().compiler().setDirective( *i_directive->asString(), i_value->forceInteger() );

   // in case of problems, an error is already raised.
}


//=========================================================
// Module

FALCON_FUNC Module_get( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );

   if( i_name == 0 || ! i_name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( ! modc->liveModule()->isAlive() )
   {
      // TODO: Find a more adequate error code.
      vm->raiseModError( new RangeError( ErrorParam( e_modver, __LINE__ ) ) );
      return;
   }

   Item *itm = modc->liveModule()->findModuleItem( *i_name->asString() );
   if( itm == 0 )
   {
      vm->raiseModError( new RangeError( ErrorParam( e_undef_sym, __LINE__ ).
         extra(*i_name->asString()) ) );
      return;
   }

   vm->retval( *itm );
}

FALCON_FUNC Module_set( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );
   Item *i_value = vm->param( 1 );

   if( i_name == 0 || ! i_name->isString() || i_value == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S,X" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( ! modc->liveModule()->isAlive() )
   {
      // TODO: Find a more adequate error code.
      vm->raiseModError( new RangeError( ErrorParam( e_modver, __LINE__ ) ) );
      return;
   }

   Item *itm = modc->liveModule()->findModuleItem( *i_name->asString() );
   if( itm == 0 )
   {
      vm->raiseModError( new RangeError( ErrorParam( e_undef_sym, __LINE__ ).
         extra(*i_name->asString()) ) );
      return;
   }

   *itm = *i_value;
}

FALCON_FUNC Module_getReference( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );

   if( i_name == 0 || ! i_name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( ! modc->liveModule()->isAlive() )
   {
      // TODO: Find a more adequate error code.
      vm->raiseModError( new RangeError( ErrorParam( e_modver, __LINE__ ) ) );
      return;
   }

   Item *itm = modc->liveModule()->findModuleItem( *i_name->asString() );
   if( itm == 0 )
   {
      vm->raiseModError( new RangeError( ErrorParam( e_undef_sym, __LINE__ ).
         extra(*i_name->asString()) ) );
      return;
   }

   vm->referenceItem( vm->regA(), *itm );
}

FALCON_FUNC Module_unload( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( ! modc->liveModule()->isAlive() )
   {
      // TODO: Find a more adequate error code.
      vm->raiseModError( new RangeError( ErrorParam( e_modver, __LINE__ ) ) );
      return;
   }

   // unlink
   if ( vm->unlink( modc->module() ) )
   {

      // destroy the reference
      delete modc;
      self->setUserData( 0 );

      // report success.
      vm->retval( (int64) 1 );
   }
   else {
      vm->retval( (int64) 0 );
   }
}

FALCON_FUNC Module_engineVersion( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );
   const Module *mod = modc->module();

   int major, minor, re;
   mod->getEngineVersion( major, minor, re );
   CoreArray *ca = new CoreArray( vm, 3 );
   ca->append( (int64) major );
   ca->append( (int64) minor );
   ca->append( (int64) re );
   vm->retval( ca );
}

FALCON_FUNC Module_moduleVersion( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );
   const Module *mod = modc->module();

   int major, minor, re;
   mod->getModuleVersion( major, minor, re );
   CoreArray *ca = new CoreArray( vm, 3 );
   ca->append( (int64) major );
   ca->append( (int64) minor );
   ca->append( (int64) re );
   vm->retval( ca );
}

}
}


/* end of compiler_ext.cpp */
