/*
   FALCON - The Falcon Programming Language.
   FILE: regex.cpp
   $Id: regex.cpp,v 1.6 2007/08/18 13:23:53 jonnymind Exp $

   Regular expression extension
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
  Regular expression extension
*/

#include <falcon/module.h>
#include <falcon/memory.h>
#include "regex_ext.h"
#include "pcre.h"
#include "version.h"

#include <stdio.h>


/** The Regex Module.

   The mdoule declares the Regex class:

   regex = Regex( pattern )
      May throw a pattern compilation error.

   regex.study()

   regex.match( str ) --> true/false
   regex.scan( str ) --> arrayOfCaptures
   regex.find( str [, startPos] ) --> first position
   regex.findAll( str [,startPos [,maxcount]] ) --> array of Positions
   regex.findAllOverlapped( str [,startPos [,maxcount]] ) --> array of Positions
   regex.replace( str, rep [, startPos ] ) --> count of replacements
   regex.replaceAll( str, rep [, repCount]] ) --> count of replacements
   regex.capturedCount()
   regex.captured()
*/

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "regex" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // Initialzie pcre -- todo, import data from app.
	pcre_malloc = Falcon::memAlloc;
   pcre_free = Falcon::memFree;
	pcre_stack_malloc = Falcon::memAlloc;
   pcre_stack_free = Falcon::memFree;

   //============================================================
   // Stub
   //
   Falcon::Symbol *regex_c = self->addClass( "Regex", Falcon::Ext::Regex_init );
   self->addClassMethod( regex_c, "study", Falcon::Ext::Regex_study );
   self->addClassMethod( regex_c, "match", Falcon::Ext::Regex_match );
   self->addClassMethod( regex_c, "grab", Falcon::Ext::Regex_grab );
   self->addClassMethod( regex_c, "find", Falcon::Ext::Regex_find );
   self->addClassMethod( regex_c, "findAll", Falcon::Ext::Regex_findAll );
   self->addClassMethod( regex_c, "findAllOverlapped", Falcon::Ext::Regex_findAllOverlapped );
   self->addClassMethod( regex_c, "replace", Falcon::Ext::Regex_replace );
   self->addClassMethod( regex_c, "replaceAll", Falcon::Ext::Regex_replaceAll );
   self->addClassMethod( regex_c, "capturedCount", Falcon::Ext::Regex_capturedCount );
   self->addClassMethod( regex_c, "captured", Falcon::Ext::Regex_captured );
   self->addClassMethod( regex_c, "compare", Falcon::Ext::Regex_compare );
   self->addClassMethod( regex_c, "version", Falcon::Ext::Regex_version );

   //==================================================
   // Error class

   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *neterr_cls = self->addClass( "RegexError", Falcon::Ext::RegexError_init );
   neterr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );


   return self;
}

/* end of regex.cpp */
