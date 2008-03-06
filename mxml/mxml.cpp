/*
   FALCON - The Falcon Programming Language.
   FILE: mxml.cpp

   The minimal XML support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 01 Mar 2008 10:23:48 +0100
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
   The mxml module - main file.
*/

#include <falcon/module.h>
#include "version.h"

#include "mxml.h"
#include "mxml_ext.h"

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "mxml" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // Class document.
   Falcon::Symbol *c_doc = self->addClass( "MXMLDocument", Falcon::Ext::MXMLDocument_init );
   self->addClassMethod( c_doc, "deserialize", Falcon::Ext::MXMLDocument_deserialize );
   self->addClassMethod( c_doc, "serialize", Falcon::Ext::MXMLDocument_serialize );
   self->addClassMethod( c_doc, "style", Falcon::Ext::MXMLDocument_style );
   self->addClassMethod( c_doc, "root", Falcon::Ext::MXMLDocument_root );
   self->addClassMethod( c_doc, "find", Falcon::Ext::MXMLDocument_find );
   self->addClassMethod( c_doc, "findPath", Falcon::Ext::MXMLDocument_findPath );
   self->addClassMethod( c_doc, "save", Falcon::Ext::MXMLDocument_save );
   self->addClassMethod( c_doc, "load", Falcon::Ext::MXMLDocument_load );
   self->addClassMethod( c_doc, "setEncoding", Falcon::Ext::MXMLDocument_setEncoding );

   Falcon::Symbol *c_nodetype = self->addClass( "MXMLStyle" );
   self->addClassProperty( c_nodetype, "MXML_STYLE_INDENT")->setInteger( MXML_STYLE_TAB );
   self->addClassProperty( c_nodetype, "MXML_STYLE_TAB" )->setInteger( MXML_STYLE_TAB );
   self->addClassProperty( c_nodetype, "MXML_STYLE_THREESPACES" )->setInteger( MXML_STYLE_THREESPACES );
   self->addClassProperty( c_nodetype, "MXML_STYLE_NOESCAPE" )->setInteger( MXML_STYLE_NOESCAPE );

   Falcon::Symbol *c_node = self->addClass( "MXMLNode", Falcon::Ext::MXMLNode_init );
   self->addClassMethod( c_node, "deserialize", Falcon::Ext::MXMLNode_deserialize );
   self->addClassMethod( c_node, "serialize", Falcon::Ext::MXMLNode_serialize );
   self->addClassMethod( c_node, "nodeType", Falcon::Ext::MXMLNode_nodeType );
   self->addClassMethod( c_node, "name", Falcon::Ext::MXMLNode_name );
   self->addClassMethod( c_node, "data", Falcon::Ext::MXMLNode_data );
   self->addClassMethod( c_node, "setAttribute", Falcon::Ext::MXMLNode_setAttribute );
   self->addClassMethod( c_node, "getAttribute", Falcon::Ext::MXMLNode_getAttribute );
   self->addClassMethod( c_node, "hasAttribute", Falcon::Ext::MXMLNode_hasAttribute );
   self->addClassMethod( c_node, "unlink", Falcon::Ext::MXMLNode_unlink );
   self->addClassMethod( c_node, "unlinkComplete", Falcon::Ext::MXMLNode_unlinkComplete );
   self->addClassMethod( c_node, "removeChild", Falcon::Ext::MXMLNode_removeChild );
   self->addClassMethod( c_node, "parent", Falcon::Ext::MXMLNode_parent );
   self->addClassMethod( c_node, "firstChild", Falcon::Ext::MXMLNode_firstChild );
   self->addClassMethod( c_node, "nextSibling", Falcon::Ext::MXMLNode_nextSibling );
   self->addClassMethod( c_node, "prevSibling", Falcon::Ext::MXMLNode_prevSibling );
   self->addClassMethod( c_node, "lastChild", Falcon::Ext::MXMLNode_lastChild );
   self->addClassMethod( c_node, "addBelow", Falcon::Ext::MXMLNode_addBelow );
   self->addClassMethod( c_node, "insertBelow", Falcon::Ext::MXMLNode_insertBelow );
   self->addClassMethod( c_node, "insertBefore", Falcon::Ext::MXMLNode_insertBefore );
   self->addClassMethod( c_node, "insertAfter", Falcon::Ext::MXMLNode_insertAfter );
   self->addClassMethod( c_node, "depth", Falcon::Ext::MXMLNode_depth );
   self->addClassMethod( c_node, "path", Falcon::Ext::MXMLNode_path );
   self->addClassMethod( c_node, "clone", Falcon::Ext::MXMLNode_clone );

   return self;
}

/* end of socket.cpp */
