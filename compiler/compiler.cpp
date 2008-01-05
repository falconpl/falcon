/*
   FALCON - The Falcon Programming Language
   FILE: compiler.cpp
   $Id: compiler.cpp,v 1.7 2007/07/27 12:03:09 jonnymind Exp $

   Compiler module main file
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
   This module exports the compiler and module loader facility to falcon
   scripts.

   In this way, Falcon scripts are able to load other modules on their own,
   or compile arbitrary code.
*/

#include <falcon/module.h>
#include "compiler_ext.h"

#include "version.h"

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "compiler" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   Falcon::Symbol *c_compiler = self->addClass( "Compiler", Falcon::Ext::Compiler_init );
   self->addClassProperty( c_compiler, "path" );
   self->addClassProperty( c_compiler, "alwaysRecomp" );
   self->addClassProperty( c_compiler, "compileInMemory" );
   self->addClassProperty( c_compiler, "ignoreSources" );
   self->addClassProperty( c_compiler, "saveModules" );
   self->addClassProperty( c_compiler, "sourceEncoding" );
   self->addClassProperty( c_compiler, "saveMandatory" );
   self->addClassProperty( c_compiler, "detectTemplate" );
   self->addClassProperty( c_compiler, "compileTemplate" );

   self->addClassMethod( c_compiler, "compile", Falcon::Ext::Compiler_compile );
   self->addClassMethod( c_compiler, "loadByName", Falcon::Ext::Compiler_loadByName );
   self->addClassMethod( c_compiler, "loadModule", Falcon::Ext::Compiler_loadModule);
   self->addClassMethod( c_compiler, "setDirective", Falcon::Ext::Compiler_setDirective);

   Falcon::Symbol *c_module = self->addClass( "Module" );
   c_module->setWKS( true );
   self->addClassProperty( c_module, "name" );
   self->addClassProperty( c_module, "path" );

   self->addClassMethod( c_module, "get", Falcon::Ext::Module_get );
   self->addClassMethod( c_module, "set", Falcon::Ext::Module_set );
   self->addClassMethod( c_module, "getReference", Falcon::Ext::Module_getReference );
   self->addClassMethod( c_module, "unload", Falcon::Ext::Module_unload );
   self->addClassMethod( c_module, "engineVersion", Falcon::Ext::Module_engineVersion );
   self->addClassMethod( c_module, "moduleVersion", Falcon::Ext::Module_moduleVersion );

   return self;
}


/* end of compiler.cpp */
