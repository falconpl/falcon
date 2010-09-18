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
#include "mxml_st.h"

/*#
   @module feather_mxml MXLM
   @brief Minimal XML support.

   This module is a very simple, fast and powerful XML parser
   and generator. It's not designed to be DOM compliant;
   W3C DOM compliancy requires some constraints that slows down
   the implementation and burden the interface. We'll provide
   a DOM compliant XML parser module in future, but for now
   we thought that a minimal, efficient and effective interface
   to handle XML documents was more important (and more helpful
   for our users) than compliancy with standard interfaces.

   In this version, the module has one important limitation:
   it is not able to self-detect the encoding of the XML document.
   It must be fed with a stream already instructed to use the
   proper encoding. Self-detection of document encoding will be
   added in the next release.

   MXML has also two major design limitations: it doesn't handle
   XML namespaces and it doesn't provide schema validation. This
   module is meant to be a very simple and slim interface to
   XML documents, and those features are rarely, if ever, needed in
   the application domain which this module is aimed to cover.

   @note In case of need, it is possible to set namespaced node name
      by including ":" in their definition. Just, MXML doesn't provide
      a separate and specific management of namespaces, while it allows
      to define and maitnain namespaces at application level.

   Apart from this limitations, the support for XML files is complete and
   features as advanced search patterns, node retrival through XML paths, and
   comment node management are provided.

   To access the functionalities of this module, load it with the instruction
   @code
      load mxml
   @endcode

   @beginmodule feather_mxml
*/

FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self


   Falcon::Module *self = new Falcon::Module();
   self->name( "mxml" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //====================================
   // Message setting
   #include "mxml_st.h"

   //=================================================================
   // Enumeration Style.
   //

   /*#
      @enum MXMLStyle
      @brief Document serialization options.

      This enumeration contains fields that can be combined through
      the OR bitwise operator (||) and that define the style that
      is used in document serialization.

      - @b INDENT: indent each node with a single space.
      - @b TAB: indent each node with a tab character (\\t).
      - @b THREESPACES: indents the nodes with three spaces.
      - @b NOESCAPE: Doesn't escape the XML characters while reading
         or writing. This is useful if the application wants to process
         escapeable sequences on its own, or if it knows that the code
         that is going to be written is not containing any escapeable
         sequence.
   */
   Falcon::Symbol *c_style = self->addClass( "MXMLStyle" );
   self->addClassProperty( c_style, "INDENT").setInteger( MXML_STYLE_INDENT ).setReadOnly( true );
   self->addClassProperty( c_style, "TAB" ).setInteger( MXML_STYLE_TAB ).setReadOnly( true );
   self->addClassProperty( c_style, "THREESPACES" ).setInteger( MXML_STYLE_THREESPACES ).setReadOnly( true );
   self->addClassProperty( c_style, "NOESCAPE" ).setInteger( MXML_STYLE_NOESCAPE ).setReadOnly( true );

   //=================================================================
   // Enumeration Node type.
   //

   /*#
      @enum MXMLType
      @brief Node types.

      This enumeration contains the types used to determine the
      apparence and significance of XML nodes.

      - tag: This node is a "standard" tag node. It's one of the declarative
         nodes which define the content of the document.
      - comment: The node contains a comment.
      - PI: The node is a "processing instruction"; a node starting with a
         question mark defines an istruction for the processor (i.e. escape
         to another language). The PI "?xml" is reserved and is not passed
         to the document parser.
      - directive: The node is a directive as i.e. DOCTYPE. Directive nodes
         start with a bang.
      - data: The node is an anonymous node containing only textual data.
      - CDATA: The node is an anonymous contains binary data
         (properly escaped as textual elements when serialized).
   */
   Falcon::Symbol *c_nodetype = self->addClass( "MXMLType" );
   self->addClassProperty( c_nodetype, "tag").setInteger( MXML::Node::typeTag ).setReadOnly( true );
   self->addClassProperty( c_nodetype, "comment" ).setInteger( MXML::Node::typeComment ).setReadOnly( true );
   self->addClassProperty( c_nodetype, "PI" ).setInteger( MXML::Node::typePI ).setReadOnly( true );
   self->addClassProperty( c_nodetype, "directive" ).setInteger( MXML::Node::typeDirective ).setReadOnly( true );
   self->addClassProperty( c_nodetype, "data" ).setInteger( MXML::Node::typeData ).setReadOnly( true );
   self->addClassProperty( c_nodetype, "CDATA" ).setInteger( MXML::Node::typeCDATA ).setReadOnly( true );

   //=================================================================
   // Enumeration error code.
   //

   /*#
      @enum MXMLErrorCode
      @brief Enumeartion listing the possible numeric error codes raised by MXML.

      This enumeration contains error codes which are set as values for the
      code field of the MXMLError raised in case of processing or I/O error.

      - @b Io: the operation couldn't be completed because of a physical error
         on the underlying stream.
      - @b Nomem: MXML couldn't allocate enough memory to complete the operation.
      - @b OutChar: Invalid characters found between tags.
      - @b InvalidNode: The node name contains invalid characters.
      - @b InvalidAtt: The attribute name contains invalid characters.
      - @b MalformedAtt: The attribute declaration doesn't conform XML standard.
      - @b InvalidChar: Character invalid in a certain context.
      - @b Unclosed: A node was open but not closed.
      - @b UnclosedEntity: The '&' entity escape was not balanced by a ';' closing it.
      - @b WrongEntity: Invalid entity name.
      - @b AttrNotFound: Searched attribute was not found.
      - @b ChildNotFound: Searched child node was not found.
      - @b Hyerarcy: Broken hierarcy; given node is not in a valid tree.
      - @b CommentInvalid: The comment node is not correctly closed by a --> sequence.
      - @b MultipleXmlDecl: the PI ?xml is declared more than once, or after another node.
   */

   Falcon::Symbol *c_errcode = self->addClass( "MXMLErrorCode" );
   self->addClassProperty( c_errcode, "Io").
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errIo );
   self->addClassProperty( c_errcode, "Nomem" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errNomem );
   self->addClassProperty( c_errcode, "OutChar" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errOutChar );
   self->addClassProperty( c_errcode, "InvalidNode" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errInvalidNode );
   self->addClassProperty( c_errcode, "InvalidAtt" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errInvalidAtt );
   self->addClassProperty( c_errcode, "MalformedAtt" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errMalformedAtt );
   self->addClassProperty( c_errcode, "InvalidChar" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errInvalidChar );
   self->addClassProperty( c_errcode, "Unclosed" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errUnclosed );
   self->addClassProperty( c_errcode, "UnclosedEntity" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errUnclosedEntity );
   self->addClassProperty( c_errcode, "WrongEntity" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errWrongEntity );
   self->addClassProperty( c_errcode, "ChildNotFound" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errChildNotFound );
   self->addClassProperty( c_errcode, "AttrNotFound" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errAttrNotFound );
   self->addClassProperty( c_errcode, "Hyerarcy" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errHyerarcy );
   self->addClassProperty( c_errcode, "CommentInvalid" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errCommentInvalid );
   self->addClassProperty( c_errcode, "MultipleXmlDecl" ).
      setInteger( FALCON_MXML_ERROR_BASE + (Falcon::int64) MXML::Error::errMultipleXmlDecl );

   //=================================================================
   // Class document.
   //
   Falcon::Symbol *c_doc = self->addClass( "MXMLDocument", Falcon::Ext::MXMLDocument_init );
   self->addClassMethod( c_doc, "deserialize", Falcon::Ext::MXMLDocument_deserialize ).asSymbol()->
      addParam("istream");
   self->addClassMethod( c_doc, "serialize", Falcon::Ext::MXMLDocument_serialize ).asSymbol()->
      addParam("ostream");
   self->addClassMethod( c_doc, "style", Falcon::Ext::MXMLDocument_style ).asSymbol()->
      addParam("setting");
   self->addClassMethod( c_doc, "root", Falcon::Ext::MXMLDocument_root );
   self->addClassMethod( c_doc, "top", Falcon::Ext::MXMLDocument_top );
   self->addClassMethod( c_doc, "find", Falcon::Ext::MXMLDocument_find ).asSymbol()->
      addParam("name")->addParam("attrib")->addParam("value")->addParam("data");
   self->addClassMethod( c_doc, "findNext", Falcon::Ext::MXMLDocument_findNext );
   self->addClassMethod( c_doc, "findPath", Falcon::Ext::MXMLDocument_findPath ).asSymbol()->
      addParam("path");
   self->addClassMethod( c_doc, "findPathNext", Falcon::Ext::MXMLDocument_findPathNext );
   self->addClassMethod( c_doc, "write", Falcon::Ext::MXMLDocument_save ).asSymbol()->
      addParam("filename");
   self->addClassMethod( c_doc, "read", Falcon::Ext::MXMLDocument_load ).asSymbol()->
      addParam("filename");
   self->addClassMethod( c_doc, "setEncoding", Falcon::Ext::MXMLDocument_setEncoding ).asSymbol()->
      addParam("encoding");
   self->addClassMethod( c_doc, "getEncoding", Falcon::Ext::MXMLDocument_getEncoding );

   //=================================================================
   // Class node
   //

   Falcon::Symbol *c_node = self->addClass( "MXMLNode", Falcon::Ext::MXMLNode_init );
   c_node->setWKS( true );
   self->addClassMethod( c_node, "deserialize", Falcon::Ext::MXMLNode_deserialize );
   self->addClassMethod( c_node, "serialize", Falcon::Ext::MXMLNode_serialize );
   self->addClassMethod( c_node, "nodeType", Falcon::Ext::MXMLNode_nodeType );
   self->addClassMethod( c_node, "name", Falcon::Ext::MXMLNode_name ).asSymbol()->
      addParam("name");
   self->addClassMethod( c_node, "data", Falcon::Ext::MXMLNode_data ).asSymbol()->
      addParam("data");
   self->addClassMethod( c_node, "setAttribute", Falcon::Ext::MXMLNode_setAttribute ).asSymbol()->
      addParam("attribute")->addParam("value");
   self->addClassMethod( c_node, "getAttribute", Falcon::Ext::MXMLNode_getAttribute ).asSymbol()->
      addParam("attribute");
   self->addClassMethod( c_node, "getAttribs", Falcon::Ext::MXMLNode_getAttribs );
   self->addClassMethod( c_node, "getChildren", Falcon::Ext::MXMLNode_getChildren );
   self->addClassMethod( c_node, "unlink", Falcon::Ext::MXMLNode_unlink );
   self->addClassMethod( c_node, "removeChild", Falcon::Ext::MXMLNode_removeChild ).asSymbol()->
      addParam("child");
   self->addClassMethod( c_node, "parent", Falcon::Ext::MXMLNode_parent );
   self->addClassMethod( c_node, "firstChild", Falcon::Ext::MXMLNode_firstChild );
   self->addClassMethod( c_node, "nextSibling", Falcon::Ext::MXMLNode_nextSibling );
   self->addClassMethod( c_node, "prevSibling", Falcon::Ext::MXMLNode_prevSibling );
   self->addClassMethod( c_node, "lastChild", Falcon::Ext::MXMLNode_lastChild );
   self->addClassMethod( c_node, "addBelow", Falcon::Ext::MXMLNode_addBelow ).asSymbol()->
      addParam("node");
   self->addClassMethod( c_node, "insertBelow", Falcon::Ext::MXMLNode_insertBelow ).asSymbol()->
      addParam("node");
   self->addClassMethod( c_node, "insertBefore", Falcon::Ext::MXMLNode_insertBefore ).asSymbol()->
      addParam("node");
   self->addClassMethod( c_node, "insertAfter", Falcon::Ext::MXMLNode_insertAfter ).asSymbol()->
      addParam("node");
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

