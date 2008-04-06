/*
   FALCON - The Falcon Programming Language.
   FILE: socket.cpp

   The configuration parser module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The confparser module - main file.
*/
#include <falcon/module.h>
#include "confparser_ext.h"
#include "confparser_mod.h"

#include "version.h"

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "confparser" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // private class socket.
   Falcon::Symbol *c_cparser = self->addClass( "ConfParser", Falcon::Ext::ConfParser_init );
   self->addClassMethod( c_cparser, "read", Falcon::Ext::ConfParser_read );
   self->addClassMethod( c_cparser, "write", Falcon::Ext::ConfParser_write );
   self->addClassMethod( c_cparser, "get", Falcon::Ext::ConfParser_get );
   self->addClassMethod( c_cparser, "getOne", Falcon::Ext::ConfParser_getOne );
   self->addClassMethod( c_cparser, "getMultiple", Falcon::Ext::ConfParser_getMultiple );
   self->addClassMethod( c_cparser, "getSections", Falcon::Ext::ConfParser_getSections );
   self->addClassMethod( c_cparser, "getKeys", Falcon::Ext::ConfParser_getKeys );
   self->addClassMethod( c_cparser, "getCategoryKeys", Falcon::Ext::ConfParser_getCategoryKeys );
   self->addClassMethod( c_cparser, "getCategory", Falcon::Ext::ConfParser_getCategory );
   self->addClassMethod( c_cparser, "removeCategory", Falcon::Ext::ConfParser_removeCategory );
   self->addClassMethod( c_cparser, "getDictionary", Falcon::Ext::ConfParser_getDictionary );
   self->addClassMethod( c_cparser, "add", Falcon::Ext::ConfParser_add );
   self->addClassMethod( c_cparser, "set", Falcon::Ext::ConfParser_set );
   self->addClassMethod( c_cparser, "remove", Falcon::Ext::ConfParser_remove );
   self->addClassMethod( c_cparser, "addSection", Falcon::Ext::ConfParser_addSection );
   self->addClassMethod( c_cparser, "removeSection", Falcon::Ext::ConfParser_removeSection );
   self->addClassMethod( c_cparser, "clearMain", Falcon::Ext::ConfParser_clearMain );
   self->addClassProperty( c_cparser, "errorLine" );
   self->addClassProperty( c_cparser, "error" );

   return self;
}

/* end of socket.cpp */
