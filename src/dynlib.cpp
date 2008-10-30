/*
   The Falcon Programming Language
   FILE: dynlib_ext.cpp

   Direct dynamic library interface for Falcon
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Main module file, providing the module object to
   the Falcon engine.
*/

#include <falcon/module.h>
#include <falcon/objectmanager.h>
#include "dynlib_ext.h"
#include "dynlib_mod.h"
#include "dynlib_st.h"

#include "version.h"

static Falcon::DynFuncManager dyn_func_manager;

/*#
   @main dynlib

   Dynamic library loader.
*/

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   #define FALCON_DECLARE_MODULE self

   // set the static engine data
   data.set();

   // initialize the module
   Falcon::Module *self = new Falcon::Module();
   self->name( "dynlib" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "dynlib_st.h"

   //============================================================
   // Here declare skeleton api
   //
   Falcon::Symbol *dynlib_cls = self->addClass( "DynLib", Falcon::Ext::DynLib_init );
   // it has no object manager, as the garbage collector doesn't handle it.
   self->addClassMethod( dynlib_cls, "get", Falcon::Ext::DynLib_get );
   self->addClassMethod( dynlib_cls, "unload", Falcon::Ext::DynLib_unload );

   Falcon::Symbol *dynfunc_cls = self->addClass( "DynFunction", Falcon::Ext::DynFunction_init );
   dynfunc_cls->getClassDef()->setObjectManager( &dyn_func_manager );
   dynfunc_cls->setWKS( true );
   self->addClassMethod( dynfunc_cls, "call", Falcon::Ext::DynFunction_call );
   self->addClassMethod( dynfunc_cls, "toString", Falcon::Ext::DynFunction_toString );

   /*#
    @class DynLibError
    @brief DynLib specific error.

    Inherited class from Error to distinguish from a standard Falcon error.
   */
   // create the base class DynLibError for falcon
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *dbierr_cls = self->addClass( "DynLibError", Falcon::Ext::DynLibError_init );
   dbierr_cls->setWKS( true );
   dbierr_cls->getClassDef()->addInheritance( new Falcon::InheritDef( error_class ) );

   return self;
}

/* end of dynlib.cpp */
