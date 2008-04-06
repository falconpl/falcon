/*
   FALCON - The Falcon Programming Language.
   FILE: mxml.cpp

   The minimal XML support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 01 Mar 2008 10:23:48 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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

   //=================================================================
   // Enumeration Style.
   //

   Falcon::Symbol *c_style = self->addClass( "MXMLStyle" );
   self->addClassProperty( c_style, "INDENT")->setInteger( MXML_STYLE_TAB );
   self->addClassProperty( c_style, "TAB" )->setInteger( MXML_STYLE_TAB );
   self->addClassProperty( c_style, "THREESPACES" )->setInteger( MXML_STYLE_THREESPACES );
   self->addClassProperty( c_style, "NOESCAPE" )->setInteger( MXML_STYLE_NOESCAPE );

   //=================================================================
   // Enumeration Node type.
   //

   Falcon::Symbol *c_nodetype = self->addClass( "MXMLType" );
   self->addClassProperty( c_nodetype, "tag")->setInteger( MXML::Node::typeTag );
   self->addClassProperty( c_nodetype, "comment" )->setInteger( MXML::Node::typeComment );
   self->addClassProperty( c_nodetype, "PI" )->setInteger( MXML::Node::typePI );
   self->addClassProperty( c_nodetype, "directive" )->setInteger( MXML::Node::typeDirective );
   self->addClassProperty( c_nodetype, "data" )->setInteger( MXML::Node::typeData );
   self->addClassProperty( c_nodetype, "CDATA" )->setInteger( MXML::Node::typeCDATA );

   //=================================================================
   // Enumeration error code.
   //

   Falcon::Symbol *c_errcode = self->addClass( "MXMLErrorCode" );
   self->addClassProperty( c_errcode, "Io")->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errIo );
   self->addClassProperty( c_errcode, "Nomem" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errNomem );
   self->addClassProperty( c_errcode, "OutChar" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errOutChar );
   self->addClassProperty( c_errcode, "InvalidNode" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errInvalidNode );
   self->addClassProperty( c_errcode, "InvalidAtt" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errInvalidAtt );
   self->addClassProperty( c_errcode, "MalformedAtt" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errMalformedAtt );
   self->addClassProperty( c_errcode, "InvalidChar" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errInvalidChar );
   self->addClassProperty( c_errcode, "Unclosed" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errUnclosed );
   self->addClassProperty( c_errcode, "UnclosedEntity" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errUnclosedEntity );
   self->addClassProperty( c_errcode, "WrongEntity" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errWrongEntity );
   self->addClassProperty( c_errcode, "MalformedAtt" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errMalformedAtt );
   self->addClassProperty( c_errcode, "ChildNotFound" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errChildNotFound );
   self->addClassProperty( c_errcode, "AttrNotFound" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errAttrNotFound );
   self->addClassProperty( c_errcode, "Hyerarcy" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errHyerarcy );
   self->addClassProperty( c_errcode, "CommentInvalid" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errCommentInvalid );
   self->addClassProperty( c_errcode, "MultipleXmlDecl" )->
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errMultipleXmlDecl );

   //=================================================================
   // Class document.
   //
   Falcon::Symbol *c_doc = self->addClass( "MXMLDocument", Falcon::Ext::MXMLDocument_init );
   self->addClassMethod( c_doc, "deserialize", Falcon::Ext::MXMLDocument_deserialize );
   self->addClassMethod( c_doc, "serialize", Falcon::Ext::MXMLDocument_serialize );
   self->addClassMethod( c_doc, "style", Falcon::Ext::MXMLDocument_style );
   self->addClassMethod( c_doc, "root", Falcon::Ext::MXMLDocument_root );
   self->addClassMethod( c_doc, "top", Falcon::Ext::MXMLDocument_top );
   self->addClassMethod( c_doc, "find", Falcon::Ext::MXMLDocument_find );
   self->addClassMethod( c_doc, "findNext", Falcon::Ext::MXMLDocument_findNext );
   self->addClassMethod( c_doc, "findPath", Falcon::Ext::MXMLDocument_findPath );
   self->addClassMethod( c_doc, "findPathNext", Falcon::Ext::MXMLDocument_findPathNext );
   self->addClassMethod( c_doc, "write", Falcon::Ext::MXMLDocument_save );
   self->addClassMethod( c_doc, "read", Falcon::Ext::MXMLDocument_load );
   self->addClassMethod( c_doc, "setEncoding", Falcon::Ext::MXMLDocument_setEncoding );
   self->addClassMethod( c_doc, "getEncoding", Falcon::Ext::MXMLDocument_getEncoding );

   //=================================================================
   // Class node
   //

   Falcon::Symbol *c_node = self->addClass( "MXMLNode", Falcon::Ext::MXMLNode_init );
   c_node->setWKS( true );
   self->addClassMethod( c_node, "deserialize", Falcon::Ext::MXMLNode_deserialize );
   self->addClassMethod( c_node, "serialize", Falcon::Ext::MXMLNode_serialize );
   self->addClassMethod( c_node, "nodeType", Falcon::Ext::MXMLNode_nodeType );
   self->addClassMethod( c_node, "name", Falcon::Ext::MXMLNode_name );
   self->addClassMethod( c_node, "data", Falcon::Ext::MXMLNode_data );
   self->addClassMethod( c_node, "setAttribute", Falcon::Ext::MXMLNode_setAttribute );
   self->addClassMethod( c_node, "getAttribute", Falcon::Ext::MXMLNode_getAttribute );
   self->addClassMethod( c_node, "getAttribs", Falcon::Ext::MXMLNode_getAttribs );
   self->addClassMethod( c_node, "getChildren", Falcon::Ext::MXMLNode_getChildren );
   self->addClassMethod( c_node, "unlink", Falcon::Ext::MXMLNode_unlink );
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

   //============================================================
   // MXML Error class
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *mxmlerr_cls = self->addClass( "MXMLError", Falcon::Ext::MXMLError_init );
   mxmlerr_cls->setWKS( true );
   mxmlerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   return self;
}

/* end of sdl.cpp */
