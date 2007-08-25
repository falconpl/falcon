/*
   FALCON - The Falcon Programming Language.
   FILE: process.cpp
   $Id: process.cpp,v 1.6 2007/07/25 12:23:08 jonnymind Exp $

   Process oriented extensions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat Jan 29 2005
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
   Process oriented extensions
*/

#include <falcon/module.h>
#include "process_mod.h"
#include "process_ext.h"

#include "version.h"

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // set the static engine data
   data.set();

   // initialize the module
   Falcon::Module *self = new Falcon::Module();
   self->name( "process" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Minimal system api
   //
   self->addExtFunc( "system", Falcon::Ext::falcon_system );
   self->addExtFunc( "systemCall", Falcon::Ext::falcon_systemCall );
   self->addExtFunc( "exec", Falcon::Ext::falcon_exec );
   self->addExtFunc( "processId", Falcon::Ext::falcon_processId );
   self->addExtFunc( "processKill", Falcon::Ext::falcon_processKill );

   //============================================================
   // Process Enumerator class
   //
   Falcon::Symbol *pe_class = self->addClass( "ProcessEnum", Falcon::Ext::ProcessEnum_init );
   self->addClassProperty( pe_class, "name" );
   self->addClassProperty( pe_class, "pid" );
   self->addClassProperty( pe_class, "parentPid" );
   self->addClassProperty( pe_class, "cmdLine" );

   self->addClassMethod( pe_class, "next", Falcon::Ext::ProcessEnum_next );
   self->addClassMethod( pe_class, "close", Falcon::Ext::ProcessEnum_close );

   //============================================================
   // Process class
   Falcon::Symbol *proc_class = self->addClass( "Process", Falcon::Ext::Process_init );
   self->addClassMethod( proc_class, "wait", Falcon::Ext::Process_wait );
   self->addClassMethod( proc_class, "close", Falcon::Ext::Process_close );
   self->addClassMethod( proc_class, "value", Falcon::Ext::Process_value );
   self->addClassMethod( proc_class, "getInput", Falcon::Ext::Process_getInput );
   self->addClassMethod( proc_class, "getOutput", Falcon::Ext::Process_getOutput );
   self->addClassMethod( proc_class, "getAux", Falcon::Ext::Process_getAux );

   //============================================================
   // ProcessError class
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *procerr_cls = self->addClass( "ProcessError", Falcon::Ext::ProcessError_init );
   procerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   //============================================================
   // Add the process attribute constants
   self->addConstant( "PROCESS_SINK_INPUT", (Falcon::int64) 0x1 );
   self->addConstant( "PROCESS_SINK_OUTPUT", (Falcon::int64) 0x2 );
   self->addConstant( "PROCESS_SINK_AUX", (Falcon::int64) 0x4 );
   self->addConstant( "PROCESS_MERGE_AUX", (Falcon::int64) 0x8 );
   self->addConstant( "PROCESS_BG", (Falcon::int64) 0x10 );
   self->addConstant( "PROCESS_USE_SHELL", (Falcon::int64) 0x20 );

   return self;
}


/* end of process.cpp */
