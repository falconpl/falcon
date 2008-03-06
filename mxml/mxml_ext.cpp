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
#include "mxml_ext.h"

#include "mxml.h"

namespace Falcon {
namespace Ext {
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
   @param style the mode in which the document is read/written
   @see read
*/

FALCON_FUNC MXMLDocument_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_style = vm->param(0);

   if ( i_style != 0 && ! i_style->isInteger() )
   {
       vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[I]" ) ) );
      return;
   }

   int style = i_style == 0 ? 0 : (int) i_style->forceInteger();

   MXML::Document *doc = new MXML::Document( style );
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
   MXML::Document *test = static_cast<MXML::Document *>( self->getUserData() );

   try
   {
      test->read( *stream );
   }
   catch( MXML::MalformedError &err )
   {
      // TODO
   }
}


FALCON_FUNC MXMLDocument_serialize( ::Falcon::VMachine *vm )
{
}

FALCON_FUNC MXMLDocument_style( ::Falcon::VMachine *vm )
{
}

FALCON_FUNC MXMLDocument_root( ::Falcon::VMachine *vm )
{
}

FALCON_FUNC MXMLDocument_find( ::Falcon::VMachine *vm )
{
}

FALCON_FUNC MXMLDocument_findPath( ::Falcon::VMachine *vm )
{
}

FALCON_FUNC MXMLDocument_save( ::Falcon::VMachine *vm )
{
}

FALCON_FUNC MXMLDocument_load( ::Falcon::VMachine *vm )
{
}




FALCON_FUNC MXMLNode_init( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_deserialize( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_serialize( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_nodeType( ::Falcon::VMachine *vm )
{
}



FALCON_FUNC MXMLNode_name( ::Falcon::VMachine *vm )
{
}



FALCON_FUNC MXMLNode_data( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_setAttribute( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_getAttribute( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_unlink( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_unlinkComplete( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_hasAttribute( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_removeChild( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_parent( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_firstChild( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_nextSibling( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_prevSibling( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_lastChild( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_addBelow( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_insertBelow( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_insertBefore( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_insertAfter( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_depth( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_path( ::Falcon::VMachine *vm )
{
}


FALCON_FUNC MXMLNode_clone( ::Falcon::VMachine *vm )
{
}





}
}

/* end of mxml_ext.cpp */
