/*
   FALCON - The Falcon Programming Language.
   FILE: mxml_ext.cpp

   Compiler module main file - extension implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Mar 2008 19:44:41 +0100
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
   MXML module main file - extension implementation.
*/

#include <falcon/vm.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include "mxml_ext.h"
#include "mxml_mod.h"

#include "mxml.h"

namespace MXML {
Falcon::CoreObject *Node::makeShell( Falcon::VMachine *vm )
{
   static Falcon::Item *node_class = 0;

   if( m_objOwner != 0 )
      return m_objOwner;

   if( node_class == 0 )
      node_class = vm->findWKI( "MXMLNode" );

   fassert( node_class != 0 );

   Falcon::CoreObject *co = node_class->asClass()->createInstance();
   co->setUserData( new Falcon::Ext::NodeCarrier( this, co ) );
   return co;
}

}

namespace Falcon {
namespace Ext {

static MXML::Node *internal_getNodeParameter( VMachine *vm, int pid )
{
   Item *i_child = vm->param(pid);

   if ( i_child == 0 || ! i_child->isObject() || ! i_child->asObject()->derivedFrom( "MXMLNode" ) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "MXMLNode" ) ) );
      return 0;
   }

   return static_cast<NodeCarrier *>( i_child->asObject()->getUserData() )->node();
}

/*# @class MXMLDocument
   Class containing a logical XML file representation.

   To work with MXML, you need to instantiate at least one object from this class;
   this represents the logic of a document. It is derived for an element, as
   an XML document must still be valid as an XML element of another document,
   but it implements some data specific for handling documents.


   The document has an (optional) style that is used on load and eventually on
   write.
*/

/*# @init MXMLDocument
   Creates the document object.
   This constructor does not load any document, and sets the style parameter to
   nothing (all defaults). Style may be changed later on with the style(int)
   method, but you need to do it before the document si read() if you want to
   set some style affects document parsing.
   @optparam encoding encoding used by the document.
   @optparam style the mode in which the document is read/written
   @see read
*/

FALCON_FUNC MXMLDocument_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_encoding = vm->param(0);
   Item *i_style = vm->param(1);

   if ( ( i_encoding != 0 && ! i_encoding->isString() && ! i_encoding->isNil() ) ||
      ( i_style != 0 && ! i_style->isInteger()) )
   {
       vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[S,I]" ) ) );
      return;
   }

   int style = i_style == 0 ? 0 : (int) i_style->forceInteger();
   MXML::Document *doc;
   if( i_encoding == 0 || i_encoding->isNil() )
      doc = new MXML::Document( "C", style );
   else
      doc = new MXML::Document( *i_encoding->asString(), style );

   self->setUserData( doc );
}


FALCON_FUNC MXMLDocument_deserialize( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_stream = vm->param(0);

   if ( i_stream == 0 || ! i_stream->isObject() || ! i_stream->asObject()->derivedFrom( "Stream" ) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "Stream" ) ) );
      return;
   }

   Stream *stream = static_cast<Stream *>( i_stream->asObject()->getUserData() );
   MXML::Document *doc = static_cast<MXML::Document *>( self->getUserData() );

   try
   {
      doc->read( *stream );
      vm->retval( true );
   }
   catch( MXML::MalformedError &err )
   {
      vm->raiseModError( new MXMLError( ErrorParam( err.numericCode(), __LINE__ )
      .desc( err.description() )
      .extra( err.describeLine() ) ) );
   }
   catch( MXML::IOError &err )
   {
      vm->raiseModError( new IoError( ErrorParam( err.numericCode(), __LINE__ )
      .desc( err.description() )
      .extra( err.describeLine() ) ) );
   }
}


FALCON_FUNC MXMLDocument_serialize( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_stream = vm->param(0);

   if ( i_stream == 0 || ! i_stream->isObject() || ! i_stream->asObject()->derivedFrom( "Stream" ) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "Stream" ) ) );
      return;
   }

   Stream *stream = static_cast<Stream *>( i_stream->asObject()->getUserData() );
   MXML::Document *doc = static_cast<MXML::Document *>( self->getUserData() );

   try
   {
      doc->write( *stream, doc->style() );
      vm->retval( true );
   }
   catch( MXML::MalformedError &err )
   {
      vm->raiseModError( new MXMLError( ErrorParam( err.numericCode(), __LINE__ )
      .desc( err.description() )
      .extra( err.describeLine() ) ) );
   }
   catch( MXML::IOError &err )
   {
      vm->raiseModError( new IoError( ErrorParam( err.numericCode(), __LINE__ )
      .desc( err.description() )
      .extra( err.describeLine() ) ) );
   }
}

FALCON_FUNC MXMLDocument_style( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_style = vm->param(0);

   if ( i_style == 0 || ! i_style->isInteger() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   MXML::Document *doc = static_cast<MXML::Document *>( self->getUserData() );
   doc->style( i_style->asInteger() );
}


FALCON_FUNC MXMLDocument_top( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Document *doc = static_cast<MXML::Document *>( self->getUserData() );
   MXML::Node *root = doc->root();
   vm->retval( root->getShell( vm ) );
}

FALCON_FUNC MXMLDocument_root( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Document *doc = static_cast<MXML::Document *>( self->getUserData() );
   MXML::Node *root = doc->main();
   // if we don't have a root (main) node, create it.
   if ( root == 0 ) {
      root = new MXML::Node( MXML::Node::typeTag, "root" );
      doc->root()->addBelow( root );
   }

   vm->retval( root->getShell( vm ) );
}

FALCON_FUNC MXMLDocument_find( ::Falcon::VMachine *vm )
{
}

FALCON_FUNC MXMLDocument_findPath( ::Falcon::VMachine *vm )
{
}

FALCON_FUNC MXMLDocument_save( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_uri = vm->param(0);

   if ( i_uri == 0 || ! i_uri->isString() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   String &uri = *i_uri->asString();
   MXML::Document *doc = static_cast<MXML::Document *>( self->getUserData() );

   //TODO: use parsing uri
   FileStream out;
   if ( out.create( uri, GenericStream::e_aUserRead | GenericStream::e_aUserWrite | GenericStream::e_aGroupRead | GenericStream::e_aOtherRead  ) )
   {
      try
      {
         doc->write( out, doc->style() );
         vm->retval( true );
      }
      catch( MXML::MalformedError &err )
      {
         vm->raiseModError( new MXMLError( ErrorParam( err.numericCode(), __LINE__ )
         .desc( err.description() )
         .extra( err.describeLine() ) ) );
      }
      catch( MXML::IOError &err )
      {
         vm->raiseModError( new IoError( ErrorParam( err.numericCode(), __LINE__ )
         .desc( err.description() )
         .extra( err.describeLine() ) ) );
      }
   }
   else
   {
      vm->raiseModError( new IoError( ErrorParam(
         FALCON_MXML_ERROR_BASE + (int) MXML::Error::errIo , __LINE__ )
         .desc( "I/O error" ) ) );
   }

   out.close();
}

FALCON_FUNC MXMLDocument_load( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_uri = vm->param(0);

   if ( i_uri == 0 || ! i_uri->isString() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S" ) ) );
      return;
   }

   String &uri = *i_uri->asString();
   MXML::Document *doc = static_cast<MXML::Document *>( self->getUserData() );

   //TODO: use parsing uri
   FileStream in;
   if ( in.open( uri ) )
   {
      try
      {
         doc->read( in );
         vm->retval( true );
      }
      catch( MXML::MalformedError &err )
      {
         vm->raiseModError( new MXMLError( ErrorParam( err.numericCode(), __LINE__ )
         .desc( err.description() )
         .extra( err.describeLine() ) ) );
      }
      catch( MXML::IOError &err )
      {
         vm->raiseModError( new IoError( ErrorParam( err.numericCode(), __LINE__ )
         .desc( err.description() )
         .extra( err.describeLine() ) ) );
      }

      in.close();
      return;
   }

   if ( ! in.good() )
   {
      vm->raiseModError( new IoError( ErrorParam(
         FALCON_MXML_ERROR_BASE + (int) MXML::Error::errIo , __LINE__ )
         .desc( "I/O error" ) ) );

   }

   in.close();
}


FALCON_FUNC MXMLDocument_setEncoding( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_encoding = vm->param(0);

   if ( i_encoding == 0 || ! i_encoding->isString() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S" ) ) );
      return;
   }

   String &encoding = *i_encoding->asString();
   Transcoder *tr = TranscoderFactory( encoding );
   if ( tr == 0 )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_param_range, __LINE__ ).
         extra( encoding ) ) );
      return;
   }
   delete tr;

   MXML::Document *doc = static_cast<MXML::Document *>( self->getUserData() );
   doc->encoding( encoding );
}

FALCON_FUNC MXMLDocument_getEncoding( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Document *doc = static_cast<MXML::Document *>( self->getUserData() );
   vm->retval( new GarbageString( vm, doc->encoding() ) );
}


/*#
   \init Node
   \brief Creates a new node
   Depending on the types the node could have a name, a data or both.

   \todo chech for name validity and throw an error
   \optparam tp one of the MXML::Node::type enum - defaults to tag
   \optparam name the name of the newborn node
   \optparam type the value of the newborn attribute
*/

FALCON_FUNC MXMLNode_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_type = vm->param(0);
   Item *i_name = vm->param(1);
   Item *i_data = vm->param(2);

   if ( ( i_type != 0 && ! i_type->isInteger() ) ||
      ( i_name != 0 && ! i_name->isString() ) ||
      ( i_data != 0 && ! i_data->isString() )  )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N,S,S]" ) ) );
      return;
   }

   // verify type range
   int type = i_type != 0 ? (int) i_type->asInteger() : 0;

   if ( type < 0 || type > (int) MXML::Node::typeFakeClosing )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "Invalid type" ) ) );
      return;
   }

   String dummy;
   String *name = i_name == 0 ? &dummy : i_name->asString();
   String *data = i_data == 0 ? &dummy : i_data->asString();

   MXML::Node *node = new MXML::Node( (MXML::Node::type) type, *name, *data );
   self->setUserData( new NodeCarrier( node, self ) );
}


FALCON_FUNC MXMLNode_serialize( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_stream = vm->param(0);

   if ( i_stream == 0 || ! i_stream->isObject() || ! i_stream->asObject()->derivedFrom( "Stream" ) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "Stream" ) ) );
      return;
   }

   Stream *stream = static_cast<Stream *>( i_stream->asObject()->getUserData() );
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();

   try
   {
      node->write( *stream, 0 );
      vm->retval( true );
   }
   catch( MXML::MalformedError &err )
   {
      vm->raiseModError( new MXMLError( ErrorParam( err.numericCode(), __LINE__ )
      .desc( err.description() )
      .extra( err.describeLine() ) ) );
   }
   catch( MXML::IOError &err )
   {
      vm->raiseModError( new MXMLError( ErrorParam( err.numericCode(), __LINE__ )
      .desc( err.description() )
      .extra( err.describeLine() ) ) );
   }

}


FALCON_FUNC MXMLNode_deserialize( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_stream = vm->param(0);

   if ( i_stream == 0 || ! i_stream->isObject() || ! i_stream->asObject()->derivedFrom( "Stream" ) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "Stream" ) ) );
      return;
   }

   Stream *stream = static_cast<Stream *>( i_stream->asObject()->getUserData() );
   delete static_cast<NodeCarrier *>( self->getUserData() );

   MXML::Node *node = new MXML::Node();

   try
   {
      node->read( *stream );
      self->setUserData( new NodeCarrier( node, self ) );
      vm->retval( self );
   }
   catch( MXML::MalformedError &err )
   {
      vm->raiseModError( new MXMLError( ErrorParam( err.numericCode(), __LINE__ )
      .desc( err.description() )
      .extra( err.describeLine() ) ) );
      delete node;
   }
   catch( MXML::IOError &err )
   {
      vm->raiseModError( new MXMLError( ErrorParam( err.numericCode(), __LINE__ )
      .desc( err.description() )
      .extra( err.describeLine() ) ) );
      delete node;
   }
}

FALCON_FUNC MXMLNode_nodeType( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();
   vm->retval( (int64)node->nodeType() );
}


FALCON_FUNC MXMLNode_name( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param(0);

   if ( i_name != 0 && ! i_name->isString() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[S]" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();
   if ( i_name == 0 )
      vm->retval( new GarbageString( vm, node->name() ) );
   else
      node->name( *i_name->asString() );

}



FALCON_FUNC MXMLNode_data( ::Falcon::VMachine *vm )
{
   Item *i_data = vm->param(0);

   if ( i_data != 0 && ! i_data->isString() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[S]" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();
   if ( i_data == 0 )
      vm->retval( new GarbageString( vm, node->data() ) );
   else
      node->data( *i_data->asString() );
}


FALCON_FUNC MXMLNode_setAttribute( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();

   Item *i_attrName = vm->param(0);
   Item *i_attrValue = vm->param(1);

   if ( i_attrName == 0 || ! i_attrName->isString() ||
        i_attrValue == 0 || ! i_attrValue->isString()
      )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S,S" ) ) );
      return;
   }

   const String &attrName = *i_attrName->asString();
   if( ! node->hasAttribute( attrName ) )
   {
      node->addAttribute( new MXML::Attribute( attrName, *i_attrValue->asString() ) );
   }

   node->setAttribute( attrName, *i_attrValue->asString() );
}


FALCON_FUNC MXMLNode_getAttribute( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();

   Item *i_attrName = vm->param(0);

   if ( i_attrName == 0 || ! i_attrName->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S" ) ) );
      return;
   }

   if ( ! node->hasAttribute( *i_attrName->asString() ) )
   {
      vm->retnil();
      return;
   }

   const String &val = node->getAttribute( *i_attrName->asString() );
   vm->retval( new GarbageString( vm, val ) );
}


FALCON_FUNC MXMLNode_unlink( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();

   node->unlink();
}

FALCON_FUNC MXMLNode_removeChild( ::Falcon::VMachine *vm )
{
   MXML::Node *child = internal_getNodeParameter( vm, 0 );
   if ( child == 0 )
      return;

   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();

   try {
      node->removeChild( child );
      vm->retval(true);
   }
   catch( ... )
   {
      vm->retval( false );
   }
}


FALCON_FUNC MXMLNode_parent( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();
   MXML::Node *parent = node->parent();
   if ( parent != 0 )
      vm->retval( parent->getShell( vm ) );
   else
      vm->retnil();
}


FALCON_FUNC MXMLNode_firstChild( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();
   MXML::Node *child = node->child();

   if ( child != 0 )
      vm->retval( child->getShell( vm ) );
   else
      vm->retnil();
}


FALCON_FUNC MXMLNode_nextSibling( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();
   MXML::Node *sibling = node->next();

   if ( sibling != 0 )
      vm->retval( sibling->getShell( vm ) );
   else
      vm->retnil();
}


FALCON_FUNC MXMLNode_prevSibling( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();
   MXML::Node *sibling = node->prev();

   if ( sibling != 0 )
      vm->retval( sibling->getShell( vm ) );
   else
      vm->retnil();
}


FALCON_FUNC MXMLNode_lastChild( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();
   MXML::Node *sibling = node->lastChild();

   if ( sibling != 0 )
      vm->retval( sibling->getShell( vm ) );
   else
      vm->retnil();
}


FALCON_FUNC MXMLNode_addBelow( ::Falcon::VMachine *vm )
{
   MXML::Node *child = internal_getNodeParameter( vm, 0 );
   if ( child == 0 )
      return;

   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();

   node->addBelow( child );
}


FALCON_FUNC MXMLNode_insertBelow( ::Falcon::VMachine *vm )
{
   MXML::Node *child = internal_getNodeParameter( vm, 0 );
   if ( child == 0 )
      return;

   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();

   node->insertBelow( child );
}


FALCON_FUNC MXMLNode_insertBefore( ::Falcon::VMachine *vm )
{
   MXML::Node *child = internal_getNodeParameter( vm, 0 );
   if ( child == 0 )
      return;

   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();

   node->insertBefore( child );
}


FALCON_FUNC MXMLNode_insertAfter( ::Falcon::VMachine *vm )
{
   MXML::Node *child = internal_getNodeParameter( vm, 0 );
   if ( child == 0 )
      return;

   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();

   node->insertAfter( child );
}


FALCON_FUNC MXMLNode_depth( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();
   vm->retval( (int64) node->depth() );
}


FALCON_FUNC MXMLNode_path( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();
   vm->retval( new GarbageString( vm, node->path() ) );
}


FALCON_FUNC MXMLNode_clone( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   MXML::Node *node = static_cast<NodeCarrier *>( self->getUserData() )->node();
   vm->retval( node->clone() );
}


FALCON_FUNC  MXMLError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new MXMLError ) );

   ::Falcon::core::Error_init( vm );
}

}
}

/* end of mxml_ext.cpp */
