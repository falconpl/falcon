/*
   FALCON - The Falcon Programming Language.
   FILE: mxml_fm.cpp

   MXML module main file - extension implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Mar 2008 19:44:41 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/native/feathers/mxml/mxml_fm.cpp"

/** \file
   MXML module main file - extension implementation.
*/

#include <falcon/vmcontext.h>
#include <falcon/stream.h>
#include <falcon/vfsiface.h>
#include <falcon/engine.h>

#include <falcon/stdhandlers.h>

#include <falcon/item.h>
#include <falcon/itemdict.h>
#include <falcon/itemarray.h>

#include "mxml_fm.h"

#include "mxml.h"

/*#
   @beginmodule mxml
*/

namespace Falcon {
namespace Feathers {

static MXML::Node *internal_getNodeParameter( Function *func, VMContext* ctx, int pid )
{
   ModuleMXML* mod = static_cast<ModuleMXML*>(func->fullModule());
   Class* clsNode = mod->classNode();
   Item *i_child = ctx->param(pid);

   if ( i_child == 0 || ! i_child->isInstanceOf(clsNode) )
   {
      throw func->paramError(__LINE__, SRC);
   }

   return static_cast<MXML::Node*>( i_child->asParentInst( clsNode ) );
}

namespace CMXMLDocument {


/*#
   @class Document
   @brief Encapsulates a complete XML document.
   @optparam encoding Encoding suggested for document load or required for document write.
   @optparam style Style required in document write.
   Class containing a logical XML file representation.

   To work with MXML, you need to instantiate at least one object from this class;
   this represents the logic of a document. It is derived for an element, as
   an XML document must still be valid as an XML element of another document,
   but it implements some data specific for handling documents.

   It is possible to specify a @b encoding parameter which must be one of the
   encoding names know by Falcon (see the @b TranscoderFactory function in the
   RTL documentation). In this version, this parameter is ignored if the object
   is used to deserialize an XML document, but it's used as output encoding (and
   set in the "encoding" field of the XML heading) when writing the document.

   The @b style parameter requires that a particular formatting is used when
   writing the document. It can be overridden in the @a MXMLDocument.write method,
   and if not provided there, the default set in the constructor will be used.

   The @b style parameter must be in @a Style enumeration.

   @note It is not necessary to create and serialize a whole XML document to
   create just XML compliant data representations. Single nodes can be serialized
   with the @a MXMLNode.serialize method; in this way it is possible to create
   small xml valid fragments for storage, network streaming, template filling
   etc. At the same time, it is possible to de-serialize a single XML node
   through the @a MXMLNode.deserialize method, which tries to decode an XML
   document fragment configuring the node and eventually re-creating its subtree.


   @section mxml_doc_struct MXML document structure.

   The XML document, as seen by the MXML module, is a tree of nodes. Some nodes have
   meta-informative value, and are meant to be used by the XML parser programs to
   determine how the tree is expected to be built.

   The tree itself has a topmost node (called top node), which is the parent for every
   other node, and a node called "root" which is actually the root of the "informative
   hierarchy" of the XML document, called 'tag nodes'.

   Tag nodes can have some "attributes", zero or more children and a partent.
   It is also possible to access the previous node at the same level of the tree,
   or the next one, and it is possible to insert nodes in any position. Tag nodes
   can have other tag nodes, data nodes or comment nodes as children. Processing Instruction
   nodes can also be placed at any level of the XML tree.

   A valid XML document can have only one root node, or in other words, it can declare
   only one tag node at top level. In example, the following is a valid XML document:
   @code
      <?xml encoding="utf-8" version="1.0" ?>
      <!-- The above was an XML special PI node, and this is a comment -->
      <!DOCTYPE greeting SYSTEM "hello.dtd">
      <!-- We see a doctype above -->
      <MyDocumentRootTag>
         ...
      </MyDocumentRootTag>
   @endcode

   In the above document, the top node would hold a comment, a DOCTYPE node, another comment
   and then a tag node, which is also the "root" node.

   The special XML instruction at the beginning is not translated into a node; instead,
   its attribute becomes readable properties of the MXMLDocument instance (or are written
   taking them from the instance properties, if the document is being written).

   Falcon MXML node allows to create automatically simple data nodes attacched to tag nodes
   by specifying a "data" value for the node. In example,
   @code
      <some_tag>Some data</some_tag>
   @endcode
   this node can be generated by creating a either a "some_tag" node and adding a data
   child to it, or setting its @a MXMLNode.data to "Some tag". Falcon will automatically
   import data for such nodes in the data field of the parent tag node.

   On the other hand, it is possible to create combination of data and
   tag children as in the following sample:
   @code
      <p>A paragraph <b>with bold text</b>.</p>
   @endcode
   In this case, it is necessary to create a "p" tag node with a child data node
   containing "A paragraph ", a tag "b" node having "with bold text" as single data and
   a data node containing a single "." character. The data in the "p" tag node will
   be empty.
*/

FALCON_DECLARE_FUNCTION( init, "encoding:[S],style:[N]" )
FALCON_DEFINE_FUNCTION_P1( init )
{
   MXML::Document* doc = static_cast<MXML::Document*>(ctx->self().asInst());

   Item *i_encoding = ctx->param(0);
   Item *i_style = ctx->param(1);

   if ( ( i_encoding != 0 && ! i_encoding->isString() && ! i_encoding->isNil() ) ||
      ( i_style != 0 && ! i_style->isInteger()) )
   {
      throw paramError( __LINE__, SRC );
   }

   int style = i_style == 0 ? 0 : (int) i_style->forceInteger();
   doc->style(style);

   if( i_encoding != 0 && ! i_encoding->isNil() )
   {
      const String& encoding = *i_encoding->asString();
      // check encoding
      if( Engine::instance()->getTranscoder(encoding) == 0 )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_range,
                  .extra("Unknown encoding \""+ encoding +"\""));
      }
      doc->encoding( *i_encoding->asString() );
   }
   else {
      doc->encoding("C");
   }

   ctx->returnFrame(ctx->self());
}


/*#
   @property style Document
   @brief Reads or changes the style applied to this XML document.

   This method allows to read or change the style used for serialization
   and deserialization of this document, which is usually set in the
   constructor.

   The @b setting parameter must be in @a Style enumeration.

   The method returns the current style as a combination of the bitfields
   from the @a Style enumeration.

   @see MXMLDocument.init
*/
static void get_style( const Class*, const String&, void* instance, Item& value )
{
   MXML::Document *doc = static_cast<MXML::Document *>( instance );
   value.setInteger( doc->style() );
}

static void set_style( const Class*, const String&, void* instance, const Item& value )
{
   MXML::Document *doc = static_cast<MXML::Document *>( instance );
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR(AccessTypeError, e_inv_prop_value, .extra("N") );
   }
   doc->style( value.forceInteger() );
}

/*#
   @property top Document
   @brief Retrieves the topmost node in the document.

   This method returns the topmost node of the document;
   this is actually an invisible node which is used as a
   "container" for the top nodes in the document: comments,
   directives as DOCTYPE and the "root" tag node.

   @see MXMLDocument.root
   @see MXMLDocument
*/
static void get_top( const Class* cls, const String&, void* instance, Item& value )
{
   MXML::Document *doc = static_cast<MXML::Document *>( instance );
   ModuleMXML* mod = static_cast<ModuleMXML*>(cls->module());
   fassert( mod != 0 );

   value = FALCON_GC_STORE( mod->classNode(), doc->root() );
}

/*#
   @method root Document
   @brief Retrieves the root tag node in the document.
   @return The root tag node of the XML document.

   This method returns the "root" node, which is the unique
   topmost node of type "tag", and that defines the information
   contents of the XML document.

   The default name for this node is "root"; the implementor
   should change the name to something more sensible before
   serializing a document generated from this instance.

   As a valid XML document must have exactly one root node,
   an instance for this node is always generated when then
   document is created, so it is always available.

   @see MXMLDocument.top
   @see MXMLDocument
*/

static void get_root( const Class* cls, const String&, void* instance, Item& value )
{
   MXML::Document *doc = static_cast<MXML::Document *>( instance );
   ModuleMXML* mod = static_cast<ModuleMXML*>(cls->module());
   fassert( mod != 0 );

   MXML::Node *root = doc->main();
   // if we don't have a root (main) node, create it.
   if ( root == 0 ) {
      root = new MXML::Node( MXML::Node::typeTag, "root" );
      doc->root()->addBelow( root );
   }

   value = FALCON_GC_STORE( mod->classNode(), root );
}

/*#
   @method find Document
   @brief Finds the first (tag) node matching a certain criterion.
   @param name Tag name of the searched node.
   @optparam attrib Name of one of the attributes of the searched node.
   @optparam value Value for one of the attributes in the node.
   @optparam data Part of the data stored in the searched tag node.
   @return The node matching the given criterion or nil if not found.

   This method performs a search in the XML tree, starting from the root,
   from a tag node with the given name, attribute (eventually having a certain
   value) and specified data portion. All the paramters are optional, and
   can be substituted with a nil or not given to match "everything".

   The @a MXMLDocument.findNext method will repeat the search starting from
   the last matching node; direction of the search is down towards the leaves
   of the tree, then forward towards the next siblings. When the nodes matching
   the criterion are exhausted, the two methods return nil.

   In example, to search in a tree for all the nodes named "interesting", the
   following code can be used:
   @code
      // doc is a MXMLDocument
      node = doc.find( "interesting" )
      while node != nil
         > "Found an interesting node:", node.path()
         ...
         node = doc.findNext()
      end
   @endcode

   To find a node which has an attribute named "cute" (at which value
   we're not interested), and which data node contains the word "suspect",
   the following code can be used:
   @code
      node = doc.find( nil, "cute", nil, "suspect" )
      while node != nil
         > "Found a suspect node:", node.path()
         ...
         node = doc.findNext()
      end
   @endcode

   @note Checks are case sensitive.

   @see MXMLDocument.findNext
*/
FALCON_DECLARE_FUNCTION( find, "name:S,attrib:[S],value:[S],data:[S]" )
FALCON_DEFINE_FUNCTION_P1( find )
{
   Item *i_name = ctx->param(0);
   Item *i_attrib = ctx->param(1);
   Item *i_valattr = ctx->param(2);
   Item *i_data = ctx->param(3);

   // parameter sanity check
   if( ( i_name == 0 || (! i_name->isString() && ! i_name->isNil() )) ||
       ( i_attrib != 0 && (! i_attrib->isString() && ! i_attrib->isNil() )) ||
       ( i_valattr != 0 && (! i_valattr->isString() && ! i_valattr->isNil() )) ||
       ( i_data != 0 && (! i_data->isString() && ! i_data->isNil() ))
   )
   {
      throw paramError( __LINE__, SRC );
   }

   String dummy;
   String *sName, *sValue, *sValAttr, *sData;

   sName = i_name == 0 || i_name->isNil() ? &dummy : i_name->asString();
   sValue = i_attrib == 0 || i_attrib->isNil() ? &dummy : i_attrib->asString();
   sValAttr = i_valattr == 0 || i_valattr->isNil() ? &dummy : i_valattr->asString();
   sData = i_data == 0 || i_data->isNil() ? &dummy : i_data->asString();

   // the real find
   MXML::Document *doc = static_cast<MXML::Document *>( ctx->self().asInst() );
   MXML::Node *node = doc->find( *sName, *sValue, *sValAttr, *sData );
   ModuleMXML* mod = static_cast<ModuleMXML*>( methodOf()->module() );

   Item ret;
   if ( node != 0 )
   {
      ret = FALCON_GC_STORE(mod->classNode(), node);
   }
   ctx->returnFrame(ret);
}


/*#
   @method findNext Document
   @brief Finds the next (tag) node matching a certain criterion.
   @return The next node matching the given criterion or nil if not found.

   This method is meant to be used after a @a MXMLDocument.find call has
   returned a valid node to iterate through all the matching nodes in a tree.

   @see MXMLDocument.find
*/
FALCON_DECLARE_FUNCTION( findNext, "" )
FALCON_DEFINE_FUNCTION_P1( findNext )
{
   MXML::Document *doc = static_cast<MXML::Document *>( ctx->self().asInst() );
   ModuleMXML* mod = static_cast<ModuleMXML*>( methodOf()->module() );
   // the real find
   MXML::Node *node = doc->findNext();

   if ( node == 0 )
      ctx->returnFrame();
   else
      ctx->returnFrame( FALCON_GC_STORE( mod->classNode(), node ) );
}

/*#
   @method findPath Document
   @brief Finds the first (tag) node matching a certain XML path.
   @param path The XML path to be searched.
   @return The next node matching the given criterion or nil if not found.

   This method provides limited (at this time, very limited) support for xpath.
   A tag node can be found through a virtual "path" staring from the root node and
   leading to it; each element of the path is a tag parent node. In example, the
   path for the node "inner" in the following example:
   @code
      <root>
         <outer>
            <middle>
               <inner>Inner content</inner>
            </middle>
         </outer>
      </root>
   @endcode

   would be "/root/outer/middle/inner".

   Paths are not unique keys for the nodes; in the above example, more than one "inner" node may
   be stacked inside the "middle" node, and all of them would have the same path.

   This method allows to use a "*" wildcard to substitute a level of the path with "anything". In
   example, the path "/root/\*\/middle/inner" would find the inner node in the above sample no matter
   what the second-topmost node name was.

   If the path cannot match any node in the three, the method returns nil. It is possible to iterate
   through all the nodes having the same path (or matching wildcard paths) in a tree by using the
   @a MXMLDocument.findPathNext method. In example, the following code would find all the nodes
   which have exactly two parents:

   @code
      node = doc.findPath( "/\*\/\*\/\*" )
      while node != nil
         > "Found a node at level 3:", node.path()
         ...
         node = doc.findPathNext()
      end
   @endcode

   @see MXMLDocument.findPathNext
*/
FALCON_DECLARE_FUNCTION( findPath, "path:[S]" )
FALCON_DEFINE_FUNCTION_P1( findPath )
{
   Item *i_name = ctx->param(0);
   MXML::Document *doc = static_cast<MXML::Document *>( ctx->self().asInst() );
   ModuleMXML* mod = static_cast<ModuleMXML*>( methodOf()->module() );

   // parameter sanity check
   if( i_name == 0 || ! i_name->isString() )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S" ) );
      return;
   }

   // the real find
   MXML::Node *node = doc->findPath( *i_name->asString() );
   if ( node == 0 )
      ctx->returnFrame();
   else
      ctx->returnFrame( FALCON_GC_STORE( mod->classNode(), node ) );
}

/*#
   @method findPathNext Document
   @brief Finds the next (tag) node matching a certain XML path.
   @return The next node matching the given criterion or nil if not found.

   This method is meant to be used together with @a MXMLDocument.findPath
   method to traverse a tree in search of nodes matching certain paths.

   @see MXMLDocument.findPath
*/
FALCON_DECLARE_FUNCTION( findPathNext, "" )
FALCON_DEFINE_FUNCTION_P1( findPathNext )
{
   MXML::Document *doc = static_cast<MXML::Document *>( ctx->self().asInst() );
   ModuleMXML* mod = static_cast<ModuleMXML*>( methodOf()->module() );

   // the real find
   MXML::Node *node = doc->findNextPath();
   if ( node == 0 )
      ctx->returnFrame();
   else
      ctx->returnFrame( FALCON_GC_STORE( mod->classNode(), node ) );
}


/*#
   @method write Document
   @brief Stores a document to an XML file on a stream.
   @param filename Name of the destination XML file.
   @optparam sytle A syle overriding the default style used by this document.
   @raise MXMLError on error during the serialization.

   This method saves the XML document to a file on a
   stream.

   @see MXMLDocument.setEncoding
*/
FALCON_DECLARE_FUNCTION( write, "stream:Stream, style:[N]" )
FALCON_DEFINE_FUNCTION_P1( write )
{
   static Class* clsStream = Engine::instance()->stdHandlers()->streamClass();

   Item *i_stream = ctx->param(0);
   Item *i_style = ctx->param(1);
   MXML::Document *doc = static_cast<MXML::Document *>( ctx->self().asInst() );

   if ( i_stream == 0 || ! i_stream->isInstanceOf(clsStream)
      || (i_style != 0 && !i_style->isOrdinal() )
   )
   {
      throw paramError(__LINE__, SRC);
   }
   
   Stream* tgt = static_cast<Stream*>(i_stream->asParentInst( clsStream ));
   Transcoder* tc = Engine::instance()->getTranscoder( doc->encoding() );
   // we have alreay filtered the transcoder when it was set.
   fassert( tc != 0 );
   int32 style = (int32) (i_style == 0 ? doc->style() : i_style->forceInteger() );
   LocalRef<TextWriter> twr( new TextWriter(tgt,tc) );
   doc->write( *twr, style );

   ctx->returnFrame();
}


/*#
   @method read MXMLDocument
   @brief Loads a document to an XML file from a stream.
   @param filename Name of the source XML file.
   @raise MXMLError on error during the deserialization.

   This method loads the XML document from a file serialized
   on a stream. The text transcoding is performed accordingly
   to the @a MXMLDocument.transcoding setting.

   @see MXMLDocument.setEncoding
*/
FALCON_DECLARE_FUNCTION( read, "stream:Stream" )
FALCON_DEFINE_FUNCTION_P1( read )
{
   static Class* clsStream = Engine::instance()->stdHandlers()->streamClass();

   Item *i_stream = ctx->param(0);

   if ( i_stream == 0 || ! i_stream->isInstanceOf(clsStream) )
   {
      throw paramError(__LINE__, SRC);
   }

   MXML::Document *doc = static_cast<MXML::Document *>( ctx->self().asInst() );
   Stream* tgt = static_cast<Stream*>(i_stream->asParentInst( clsStream ));
   Transcoder* tc = Engine::instance()->getTranscoder( doc->encoding() );
   // we have alreay filtered the transcoder when it was set.
   fassert( tc != 0 );
   LocalRef<TextReader> twr( new TextReader(tgt,tc));
   doc->read( *twr );

   ctx->returnFrame();
}

/*#
   @proprty encoding Document
   @brief Document encoding for stream operations.
   @raise ParamError if the setting an unknown encoding name.

   This method sets the encoding used for I/O operations on this
   XML document. It also determines the value of the "encoding"
   attribute that will be set in the the special PI ?xml at
   document heading.
*/

static void get_encoding( const Class*, const String&, void* instance, Item& value )
{
   MXML::Document *doc = static_cast<MXML::Document *>( instance );
   value = FALCON_GC_HANDLE( new String(doc->encoding()) );
}

static void set_encoding( const Class*, const String&, void* instance, const Item& value )
{
   MXML::Document *doc = static_cast<MXML::Document *>( instance );

   if( ! value.isString() )
   {
      throw FALCON_SIGN_XERROR(AccessTypeError, e_inv_prop_value, .extra("N") );
   }

   const String& encoding = *value.asString();
   if( Engine::instance()->getTranscoder( encoding ) == 0 )
   {
      throw FALCON_SIGN_XERROR(AccessTypeError, e_inv_prop_value, .extra("Unknown encoding \"" + encoding + "\"" ) );
   }

   doc->encoding( encoding );
}


} // end of namespace CMXMLDocument


ClassDocument::ClassDocument():
         Class("Document")
{
   setConstuctor( new CMXMLDocument::Function_init );

   addProperty("encoding", &CMXMLDocument::get_encoding, &CMXMLDocument::set_encoding );
   addProperty("root", &CMXMLDocument::get_root );
   addProperty("top", &CMXMLDocument::get_top );
   addProperty("style", &CMXMLDocument::get_style, &CMXMLDocument::set_style );

   addMethod( new CMXMLDocument::Function_find );
   addMethod( new CMXMLDocument::Function_findNext );
   addMethod( new CMXMLDocument::Function_findPath );
   addMethod( new CMXMLDocument::Function_findPathNext );
   addMethod( new CMXMLDocument::Function_read );
   addMethod( new CMXMLDocument::Function_write );
}

ClassDocument::~ClassDocument()
{

}

int64 ClassDocument::occupiedMemory( void* ) const
{
   // actually should be the size of the sub-tree
   return sizeof(MXML::Node) + 16;
}

void* ClassDocument::createInstance() const
{
   return new MXML::Document("C");
}

void ClassDocument::dispose( void* instance ) const
{
   // while there is 1 reference to any node, or to the doc
   // there can't be any dispose invoked.
   // When dispose on a node or on the document is invoked,
   // all the tree must be disposed at once.
   MXML::Document* doc = static_cast<MXML::Document*>(instance);
   delete doc;
}

void* ClassDocument::clone( void* instance ) const
{
   MXML::Document* doc = static_cast<MXML::Document*>(instance);
   MXML::Document* copy = new MXML::Document(*doc);
   return copy;
}

void ClassDocument::gcMarkInstance( void* instance, uint32 mark ) const
{
   MXML::Document* doc = static_cast<MXML::Document*>(instance);
   doc->gcMark(mark);
}

bool ClassDocument::gcCheckInstance( void* instance, uint32 mark ) const
{
   MXML::Document* doc = static_cast<MXML::Document*>(instance);
   return doc->currentMark() >= mark;
}

//===============================================================================
// MXML NODE
//===============================================================================


/*#
   @class Node
   @optparam type One of the node type defined by the @a Type enumeration.
   @optparam name Name of the node, if this is a tag node.
   @optparam data Optional data content attached to this node..
   @brief Minimal entity of the XML document.
   @raise ParamError if the type is invalid.

   This class encapsulates a minimal addressable entity in an XML document.
   Nodes can be of different types, some of which, like CDATA, tag and comment nodes
   can have a simple textual data attached to them (equivalent to a single data node
   being their only child).

   Nodes can be attached and detached from trees or serialized on their own. The
   subtrees of child nodes stays attached to its parent also when the MXMLDocument
   they are attached to is changed. Also, serializing a node directly allows to
   write mini xml valid fragments which may be used for network transmissions,
   database storage, template filling etc., without the need to build a whole XML
   document and writing the ?xml heading declarator.

   The @b type must be one of the @a Type enumeration elements.
   The @b name of the node is relevant only for Processing Instruction
   nodes and tag node, while data can be specified for comment, tag and
   data nodes.

   If the node is created just to be de-serialized, create it as an empty
   comment and then deserialize the node from a stream.

*/

namespace CMXMLNode {

FALCON_DECLARE_FUNCTION( init, "type:[N],name:[S],data:[S]" )
FALCON_DEFINE_FUNCTION_P1( init )
{
   Item *i_type = ctx->param(0);
   Item *i_name = ctx->param(1);
   Item *i_data = ctx->param(2);

   MXML::Node* node = static_cast<MXML::Node*>( ctx->self().asInst() );

   if ( ( i_type != 0 && ! i_type->isInteger() ) ||
      ( i_name != 0 && ! (i_name->isString() || i_name->isNil()) ) ||
      ( i_data != 0 && ! i_data->isString() )  )
   {
      throw paramError(__LINE__, SRC);
   }

   // verify type range
   int type = i_type != 0 ? (int) i_type->asInteger() : ((int)MXML::Node::typeComment) ;

   if ( type < 0 || type > (int) MXML::Node::typeFakeClosing )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra( "Invalid type" ) );
   }

   node->nodeType((MXML::Node::type) type);
   if( i_name != 0 )
   {
      node->name( *i_name->asString() );
   }

   if( i_data != 0 )
   {
      node->data( *i_data->asString() );
   }
   ctx->returnFrame(ctx->self());
}

static void internal_read_write( Function* func, VMContext* ctx, bool bRead )
{
   static Class* clsStream = Engine::instance()->stdHandlers()->streamClass();

   Item *i_stream = ctx->param(0);
   Item *i_encoding = ctx->param(1);

   if ( i_stream == 0 || ! i_stream->isInstanceOf(clsStream)
     || (i_encoding != 0 && ! i_encoding->isString() )
   )
   {
      throw func->paramError( __LINE__, SRC );
   }

   Stream *stream = static_cast<Stream *>( i_stream->asParentInst( clsStream ) );
   MXML::Node *node = static_cast< MXML::Node *>( ctx->self().asInst() );

   String encoding = i_encoding == 0 ? "C" : *i_encoding->asString();
   Transcoder* tc = Engine::instance()->getTranscoder( encoding );
   if( tc == 0 )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_param_range, .extra("Unknown encoding \"" + encoding + "\"") );
   }

   try
   {
      if( bRead )
      {
         LocalRef<TextReader> tr( new TextReader(stream) );
         node->read( *tr, 0 );
      }
      else {
         LocalRef<TextWriter> tw( new TextWriter(stream) );
         node->write( *tw, 0 );
      }
   }
   catch( MXML::MalformedError &err )
   {
      throw new MXMLError(
        ErrorParam( FALCON_MXML_ERROR_BASE + err.numericCode(), __LINE__, SRC )
         .desc( err.description() )
         .extra( err.describeLine() ) );
   }

   ctx->returnFrame();
}

/*#
   @method write MXMLNode
   @brief Stores an XML node on a stream.
   @param stream The stream where to store the node.
   @optparam encoding The name of the text encoding to be used.

   This method allows the storage of a single node and all its
   children in an XML format. The resulting data
   is an valid XML fragment that may be included verbatim in
   an XML document.

   If not given, the encoding will be "C" (untranslated)
*/
FALCON_DECLARE_FUNCTION( write, "stream:Stream,encoding:[S]" )
FALCON_DEFINE_FUNCTION_P1( write )
{
   internal_read_write( this, ctx, false );
}

/*#
   @method read MXMLNode
   @brief Stores an XML node on a stream.
   @param stream The stream from which to read the node.
   @optparam encoding The name of the text encoding to be used.
   @raise MXMLError If the deseerialization failed.

   This method allows the storage of a single node and all its
   children in an XML compliant format. The resulting data
   is an valid XML fragment that may be included verbatim in
   an XML document.
*/
FALCON_DECLARE_FUNCTION( read, "stream:Stream,encoding:[S]" )
FALCON_DEFINE_FUNCTION_P1( read )
{
   internal_read_write( this, ctx, true );
}


/*#
   @property type Node
   @brief the type of this node.

   This property can be used to change the type of a node once it has been created.
*/

static void get_type( const Class*, const String&, void* instance, Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   value.setInteger( node->nodeType() );
}

static void set_type( const Class*, const String&, void* instance, const Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra("N") );
   }

   int type = value.asInteger();
   if ( type < 0 || type > (int) MXML::Node::typeFakeClosing )
   {
      throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra( "Invalid XML node type" ) );
   }

   node->nodeType(static_cast<MXML::Node::type>(type));
}


/*#
   @property name Node
   @brief Set and/or return the name of this node.

   A name can be assigned to any node, but it will be meaningful only
   for tag and PI nodes.

   The name assigned to a node cannot be empty.

   @note Each access to this properties creates a new mutable copy of the
   string representing the name of this node. If used repeatedly, it's advisable
   to cache the node name in a local variable.
*/
static void get_name( const Class*, const String&, void* instance, Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   value = FALCON_GC_HANDLE( new String( node->name()) );
}

static void set_name( const Class*, const String&, void* instance, const Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   if( ! value.isString() )
   {
      throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra("N") );
   }

   const String& name = *value.asString();
   if ( name.empty() )
   {
      throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra( "Invalid XML node name" ) );
   }

   node->name( name );
}


/*#
   @proprty data Node
   @brief Set and/or return the content of this node.
   @optparam data If provided, the new data of this node.
   @return If a new @b data is not given, the current node data.

   A data can be assigned to any node, but it will be meaningful only
   for data, tag, comment and CDATA nodes. Moreover, tag nodes can have
   also other children; in this case, the data set with this method will
   be serialized as if it was a first child data node.

   @note Each access to this properties creates a new mutable copy of the
   string representing the data of this node. If used repeatedly, it's advisable
   to cache the node name in a local variable.
*/
static void get_data( const Class*, const String&, void* instance, Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   value = FALCON_GC_HANDLE( new String( node->data()) );
}

static void set_data( const Class*, const String&, void* instance, const Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   if( ! value.isString() )
   {
      throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra("N") );
   }

   const String& data = *value.asString();
   node->data( data );
}

/*#
   @method setAttribute Node
   @brief Sets an XML attribute of this node to a given value.
   @param attribute The XML attribute to be set.
   @param value The value for this XML attribute (as a string).

   This method sets the value for an XML attribute of the node.
   Attributes can be assigned to PI, Tag and DOCTYPE nodes.

   The @b value parameter can be any Falcon type; if it's not
   a string, the @b FBOM.toString method will be applied to transform
   it into a string.

   If the attribute doesn't exist, it is added, otherwise it's value
   is changed.

   @note Don't confuse XML attributes with Falcon attributes.
*/
FALCON_DECLARE_FUNCTION( setAttribute, "attribute:S,value:S" )
FALCON_DEFINE_FUNCTION_P1( setAttribute )
{
   Item *i_attrName = ctx->param(0);
   Item *i_attrValue = ctx->param(1);

   if ( i_attrName == 0 || ! i_attrName->isString() ||
        i_attrValue == 0 || ! i_attrName->isString()
      )
   {
      throw paramError(__LINE__, SRC);
   }

   const String& value = * i_attrValue->asString();
   const String& attrName = *i_attrName->asString();
   MXML::Node *node = static_cast<MXML::Node *>( ctx->self().asInst() );
   if( ! node->hasAttribute( attrName ) )
   {
      node->addAttribute( new MXML::Attribute( attrName, value ) );
   }
   else {
      node->setAttribute( attrName, value );
   }

   ctx->returnFrame();
}

/*#
   @method getAttribute Node
   @brief Gets the value of an XML attribute of this node.
   @param attribute The XML attribute to be read.
   @return The value for this XML attribute (as a string).

   This method retrieves the value for an XML attribute of the node.
   Attributes can be assigned to PI, Tag and DOCTYPE nodes.

   If the attribute doesn't exist, nil is returned.

   @note Don't confuse XML attributes with Falcon attributes.
*/
FALCON_DECLARE_FUNCTION( getAttribute, "attribute:S,value:S" )
FALCON_DEFINE_FUNCTION_P1( getAttribute )
{
   Item *i_attrName = ctx->param(0);

   if ( i_attrName == 0 || ! i_attrName->isString() )
   {
      throw paramError(__LINE__, SRC);
   }

   const String& attrName = *i_attrName->asString();
   MXML::Node *node = static_cast<MXML::Node *>( ctx->self().asInst() );
   if( node->hasAttribute( attrName ) )
   {
      const String& value = node->getAttribute(attrName);
      ctx->returnFrame( FALCON_GC_HANDLE( new String(value) )  );
   }
   else {
      ctx->returnFrame();
   }
}

/*#
   @property attributes Node
   @brief Gets the all the XML attributes of this node.
   @return A dictionary containing all the XML attributes and their values.

   This method retrieves all the attributes of the node, and stores them
   in a dictionary as a pair of key => value strings.

   Attributes can be assigned to PI, Tag and DOCTYPE nodes.

   If the node doesn't have any XML attribute, an empty dictionary is
   returned.

   The dictionary is read-only; values in the dictionary can be changed,
   but this won't change the values of the original XML attributes in
   the source node.

   @note Don't confuse XML attributes with Falcon attributes, accessed via the
   @a BOM.attribs proeprty.
*/
static void get_attributes( const Class*, const String&, void* instance, Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );

   const MXML::AttribList &attribs = node->attribs();
   MXML::AttribList::const_iterator iter = attribs.begin();

   ItemDict* dict = new ItemDict;
   while( iter != attribs.end() )
   {
      const String& name = (*iter)->name();
      const String& value = (*iter)->value();
      dict->insert( FALCON_GC_HANDLE(new String(name)), FALCON_GC_HANDLE(new String(value)) );
      ++iter;
   }

   value = FALCON_GC_HANDLE(dict);
}

static void set_attributes( const Class*, const String&, void* instance, const Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   if( ! value.isDict() )
   {
      throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra("D:S=>S") );
   }

   class Rator: public ItemDict::Enumerator
   {
   public:
      Rator( MXML::Node* node ): m_node(node) {}
      virtual ~Rator() {}
      virtual void operator()( const Item& key, Item& value )
      {
         if (! key.isString() || ! value.isString() )
         {
            throw FALCON_SIGN_XERROR( AccessTypeError, e_inv_prop_value, .extra("D:S=>S") );
         }

         const String& skey = *key.asString();
         const String& svalue = *value.asString();

         if( ! m_node->hasAttribute( skey ) )
         {
            m_node->addAttribute( new MXML::Attribute( skey, svalue ) );
         }
         else {
            m_node->setAttribute( skey, svalue );
         }
      }
   private:
      MXML::Node* m_node;
   };

   Rator rator(node);
   ItemDict* dict = value.asDict();
   dict->enumerate( rator );
}

/*#
   @property children Node
   @brief Gets the all the children nodes of this node.

   This method stores all the children of an XML node in an
   array.

   If the node doesn't have any child, an empty array is
   returned.

   The array is read-only; it is possible to change it but
   inserting or removing nodes from it won't change the children
   list of the source node.

   @note This property is read-only.
*/

static void get_children( const Class* clsNode, const String&, void* instance, Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );

   ItemArray* arr = new ItemArray;
   node = node->child();
   while( node != 0 )
   {
      arr->append( FALCON_GC_STORE( clsNode, node ) );
      node = node->next();
   }

   value = FALCON_GC_HANDLE(arr);
}


/*#
   @method unlink Node
   @brief Removes a node from its parent tree.

   This method removes a node from the list of node of
   its parent node. The node is removed together with all
   its children and their whole subtree.

   After an unlink, it is possible to insert the node into
   another place of the same tree or of another tree.

   Actually, all the insertion routines perform an @b unlink on
   the node that is going to be inserted, so it is not
   necessary to call @b unlink from the falcon script before
   adding it elsewhere. However, explicitly unlinked node may be
   kept elsewhere (i.e. in a script maintained dictionary) for
   later usage.
*/
FALCON_DECLARE_FUNCTION( unlink, "" )
FALCON_DEFINE_FUNCTION_P1( unlink )
{
   MXML::Node *node = static_cast<MXML::Node *>( ctx->self().asInst() );
   node->unlink();
   ctx->returnFrame();
}

/*#
   @method removeChild Node
   @brief Removes a child from its parent tree.
   @param child The child node to be removed.
   @return True if the @b child node is actually a child of this node, false otherwise.

   This method is equivalent to @b MXMLNode.unlink applied to the child node,
   but it checks if the removed node is really a child of this node before actually
   removing it.

   If the @b child parameter is really a child of this node it is unlinked and the
   method returns true, otherwise the node is untouched and the method returns false.
*/

FALCON_DECLARE_FUNCTION( removeChild, "child:Node" )
FALCON_DEFINE_FUNCTION_P1( removeChild )
{
   MXML::Node *child = internal_getNodeParameter( this, ctx, 0 );
   MXML::Node *node = static_cast<MXML::Node *>( ctx->self().asInst() );

   try {
      node->removeChild( child );
      ctx->returnFrame(Item().setBoolean(true));
   }
   catch( ... )
   {
      ctx->returnFrame(Item().setBoolean(false));
   }
   ctx->returnFrame();
}

/*#
   @property parent Node
   @brief Return the parent node of this node.

   This property holds the node that is currently
   parent of this node in the XML tree.

   The property returns nil if the node hasn't a parent; this may mean
   that this node is the topmost node in an XMLDocument (the node
   returned by @a Document.top ) or if it has not still been added
   to a tree, or if it has been removed with @a MXMLNode.removeChild or
   @a Node.unlink.

   @note This property is read-only
*/
static void get_parent( const Class* cls, const String&, void* instance, Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   if( node->parent() != 0 )
   {
      value = FALCON_GC_STORE( cls, node->parent() );
   }
   else {
      value.setNil();
   }
}


/*#
   @property firstChild Node
   @brief Return the first child of a node.

   This property returns the first child of a node; it's the node that will
   be delivered for first in the rendering of the final XML document, and that
   will appear on the topmost position between the nodes below the current
   one.

   To iterate through all the child nodes of a node, it is possible to
   get the first child and the iteratively @a MXMLNode.nextSibling
   until it returns nil. In example:

   @code
      // node is an Node...
      child = node.firstChild
      while child != nil
         > "Child of ", node.name(), ": ", child.name()
         child = child.nextSibling
      end
   @endcode

   @note This property is read-only
   @see Node.insertBelow
*/
static void get_firstChild( const Class* cls, const String&, void* instance, Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   if( node->child() != 0 )
   {
      value = FALCON_GC_STORE( cls, node->child() );
   }
   else {
      value.setNil();
   }
}

/*#
   @property nextSibling MXMLNode
   @brief Return the next node child of the same parent.
   @return The next node at the same level, or nil.

   This method returns the next node that would be found in
   the rendered XML document right after this one, at the same level.
   If such node doesn't exist, it returns nil.

   @see Node.firstChild
   @note This property is read-only
*/
static void get_nextSibling( const Class* cls, const String&, void* instance, Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   if( node->next() != 0 )
   {
      value = FALCON_GC_STORE( cls, node->next() );
   }
   else {
      value.setNil();
   }
}

/*#
   @property prevSibling Node
   @brief Return the previous node child of the same parent.

   This method returns the previous node that would be found in
   the rendered XML document right after this one, at the same level.
   If such node doesn't exist, it returns nil.

   @see Node.lastChild
   @note This property is read-only
*/
static void get_prevSibling( const Class* cls, const String&, void* instance, Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   if( node->prev() != 0 )
   {
      value = FALCON_GC_STORE( cls, node->prev() );
   }
   else {
      value.setNil();
   }
}

/*#
   @property lastChild MXMLNode
   @brief Return the last child of a node.

   This method returns the last child of a node; it's the node that will
   be delivered for last in the rendering of the final XML document, and that
   will appear on the lowest position between the nodes below the current
   one.

   To iterate through all the child nodes of a node in reverse order,
   it is possible to get the last child and the iteratively
   @a Node.prevSibling
   until it returns nil. For example:

   @code
      // node is an MXMLNode...
      child = node.lastChild
      while child != nil
         > "Child of ", node.name(), " reverse: ", child.name()
         child = child.prevSibling
      end
   @endcode
*/

static void get_lastChild( const Class* cls, const String&, void* instance, Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   if( node->lastChild() != 0 )
   {
      value = FALCON_GC_STORE( cls, node->lastChild() );
   }
   else {
      value.setNil();
   }
}

/*#
   @method addBelow Node
   @brief Adds a node below this one.
   @param node The node to be added below this one.

   This method appends the given @b node as the last child
   of this node, eventually removing it from a previous tree
   structure to which it was linked if needed.

   After this method returns, @b node can be retrieved calling the
   @a Node.lastChild on this node, until another @b addBelow
   adds another node at the end of the children list.
*/
FALCON_DECLARE_FUNCTION( addBelow, "child:Node" )
FALCON_DEFINE_FUNCTION_P1( addBelow )
{
   MXML::Node *child = internal_getNodeParameter( this, ctx, 0 );
   MXML::Node *node = static_cast<MXML::Node *>( ctx->self().asInst() );

   // just to be sure
   child->unlink();
   node->addBelow( child );
   ctx->returnFrame();
}

/*#
   @method prependBelow Node
   @brief Inserts a node below this one.
   @param node The node to be added below this one.

   This method prepends the given @b node as the first child
   of this node, eventually removing it from a previous tree
   structure to which it was linked if needed.

   After this method returns, @b node can be retrieved calling the
   @a Node.firstChild on this node, until another @b insertBelow
   adds another node at the beginning of the children list.
*/
FALCON_DECLARE_FUNCTION( prependBelow, "child:Node" )
FALCON_DEFINE_FUNCTION_P1( prependBelow )
{
   MXML::Node *child = internal_getNodeParameter( this, ctx, 0 );
   MXML::Node *node = static_cast<MXML::Node *>( ctx->self().asInst() );

   // just to be sure
   child->unlink();
   node->insertBelow( child );
}

/*#
   @method insertBefore Node
   @brief Inserts a node before this one.
   @param node The node to be added before this one.

   This method prepends the given @b node in front of this one
   in the list of sibling nodes, eventually removing it from a previous tree
   structure to which it was linked if needed. This is equivalent to inserting
   the node exactly before this one, at the same level, in the final
   XML document.

   If this node was the first child of its parent, the inserted node
   becomes the new first child.
*/
FALCON_DECLARE_FUNCTION( insertBefore, "child:Node" )
FALCON_DEFINE_FUNCTION_P1( insertBefore )
{
   MXML::Node *child = internal_getNodeParameter( this, ctx, 0 );
   MXML::Node *node = static_cast<MXML::Node *>( ctx->self().asInst() );
   // just to be sure
   child->unlink();
   node->insertBefore( child );
   ctx->returnFrame();
}


/*#
   @method insertAfter Node
   @brief Inserts a node after this one.
   @param node The node to be added after this one.

   This method prepends the given @b node after of this one
   in the list of sibling nodes, eventually removing it from a previous tree
   structure to which it was linked if needed. This is equivalent to inserting
   the node exactly after this one, at the same level, in the final
   XML document.

   If this node was the last child of its parent, the inserted node
   becomes the new last child.
*/
FALCON_DECLARE_FUNCTION( insertAfter, "child:Node" )
FALCON_DEFINE_FUNCTION_P1( insertAfter )
{
   MXML::Node *child = internal_getNodeParameter( this, ctx, 0 );
   MXML::Node *node = static_cast<MXML::Node *>( ctx->self().asInst() );

   // just to be sure
   child->unlink();
   node->insertAfter( child );
   ctx->returnFrame();
}

/*#
   @property depth Node
   @brief Calculates the depth of this node.

   This property returns the number of steps needed to find a
   node without parents in the parent hierarchy of this node.

   The depth for a topmost tree node is 0, for a root node in a tree
   is 1 and for its direct child is 2.
*/
static void get_depth( const Class*, const String&, void* instance, Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   value.setInteger( (int64) node->depth() );
}


/*#
   @property path Node
   @brief The path from the root to this node.

   The path of a node is the list of parent node names separated
   by a slash "/", starting from the root node (or from the first
   node of a separate tree) and terminating with the node itself.

   In example, the path of the node "item" in the following XML document:
   @code
      <root>
         <content>
            <item/>
         </content>
      </root>
   @endcode
   would be "/root/content/item"

   @see MXMLDocument.findPath
*/
static void get_path( const Class*, const String&, void* instance, Item& value )
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   value = FALCON_GC_HANDLE( new String(node->path()) );
}

/*#
   @method clone Node
   @brief Clones a whole XML hierarchy starting from this node.
   @return A copy of this node, with all its children copied.
*/
}


ClassNode::ClassNode():
         Class("Node")
{
   m_bHasSharedInstances = true;
   setConstuctor( new CMXMLNode::Function_init );

   addProperty("attributes", &CMXMLNode::get_attributes, &CMXMLNode::set_attributes );
   addProperty("name", &CMXMLNode::get_name, &CMXMLNode::set_name );
   addProperty("data", &CMXMLNode::get_data, &CMXMLNode::set_data );
   addProperty("type", &CMXMLNode::get_type, &CMXMLNode::set_type );
   addProperty("children", &CMXMLNode::get_children );
   addProperty("firstChild", &CMXMLNode::get_firstChild );
   addProperty("lastChild", &CMXMLNode::get_lastChild );
   addProperty("prevSibling", &CMXMLNode::get_prevSibling );
   addProperty("nextSibling", &CMXMLNode::get_nextSibling );
   addProperty("depth", &CMXMLNode::get_depth );
   addProperty("parent", &CMXMLNode::get_parent );
   addProperty("path", &CMXMLNode::get_path );

   addMethod( new CMXMLNode::Function_addBelow );
   addMethod( new CMXMLNode::Function_prependBelow );
   addMethod( new CMXMLNode::Function_insertAfter );
   addMethod( new CMXMLNode::Function_insertBefore );
   addMethod( new CMXMLNode::Function_removeChild );
   addMethod( new CMXMLNode::Function_unlink );

   addMethod( new CMXMLNode::Function_getAttribute );
   addMethod( new CMXMLNode::Function_setAttribute );

   addMethod( new CMXMLNode::Function_read );
   addMethod( new CMXMLNode::Function_write );

}

ClassNode::~ClassNode()
{

}

int64 ClassNode::occupiedMemory( void* ) const
{
   // actually should be the size of the sub-tree
   return sizeof(MXML::Node) + 16;
}

void* ClassNode::createInstance() const
{
   return new MXML::Node;
}

void ClassNode::dispose( void* instance ) const
{
   // When dispose on a node or on the document is invoked,
   // all the tree must be disposed at once.

   MXML::Node* node = static_cast<MXML::Node*>(instance);
   // this finds the topmost node and destroys it.
   node->dispose();
}

void* ClassNode::clone( void* instance ) const
{
   MXML::Node *node = static_cast<MXML::Node *>( instance );
   return node->clone();
}

void ClassNode::gcMarkInstance( void* instance, uint32 mark ) const
{
   MXML::Node* node = static_cast<MXML::Node*>(instance);
   node->gcMark(mark);
}

bool ClassNode::gcCheckInstance( void* instance, uint32 mark ) const
{
   MXML::Node* node = static_cast<MXML::Node*>(instance);
   return node->currentMark() >= mark;
}


//=======================================================================
// MXML error class
//

/*#
   @class MXMLError
   @brief Error raised by the MXML module in case of problems.
   @optparam code A numeric error code.
   @optparam description A textual description of the error code.
   @optparam extra A descriptive message explaining the error conditions.
   @from Error code, description, extra

   An instance of this class is raised whenever some problem is
   found. The error codes generated by this module are in the
   @a ErrorCode enumeration.
*/

//=======================================================================
// MXML module class
//

Enum::Enum( const String& name ):
         Class(name)
{}

Enum::~Enum()
{}

void* Enum::createInstance() const
{
   return 0;
}

void Enum::dispose( void* ) const
{
   // do nothing
}

void* Enum::clone( void* ) const
{
   return 0;
}

/*#
  @enum Style
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
ClassStyle::ClassStyle():
         Enum("Style")
{
   addConstant("INDENT", MXML_STYLE_INDENT);
   addConstant("TAB", MXML_STYLE_TAB);
   addConstant("THREESPACES", MXML_STYLE_THREESPACES);
   addConstant("NOESCAPE", MXML_STYLE_NOESCAPE);
}

ClassStyle::~ClassStyle()
{}

/*#
    @enum NodeType
    @brief Node types.

    This enumeration contains the types used to determine the
    appearance and significance of XML nodes.

    - tag: This node is a "standard" tag node. It's one of the declarative
       nodes which define the content of the document.
    - comment: The node contains a comment.
    - PI: The node is a "processing instruction"; a node starting with a
       question mark defines an instruction for the processor (i.e. escape
       to another language). The PI "?xml" is reserved and is not passed
       to the document parser.
    - directive: The node is a directive as i.e. DOCTYPE. Directive nodes
       start with a bang.
    - data: The node is an anonymous node containing only textual data.
    - CDATA: The node is an anonymous contains binary data
       (properly escaped as textual elements when serialized).
    - XMLDECL xml declaration ("?xml") on top of an XML document.
 */
ClassNodeType::ClassNodeType():
         Enum("NodeType")
{
   addConstant("comment", (int) MXML::Node::typeComment );
   addConstant("CDATA", (int) MXML::Node::typeCDATA );
   addConstant("data", (int) MXML::Node::typeData );
   addConstant("directive", (int) MXML::Node::typeDirective );
   addConstant("DOCUMENT", (int) MXML::Node::typeDocument );
   addConstant("XMLDECL", (int) MXML::Node::typeXMLDecl );
   addConstant("tag", (int) MXML::Node::typeTag );
   addConstant("PI", (int) MXML::Node::typePI );
}

ClassNodeType::~ClassNodeType()
{}

/*#
   @enum ErrorCode
   @brief Enumeration listing the possible numeric error codes raised by MXML.

   This enumeration contains error codes which are set as values for the
   code field of the MXMLError raised in case of processing or I/O error.

   - @b IO: the operation couldn't be completed because of a physical error
      on the underlying stream (in case the Stream class didn't throw an
      exception itself).
   - @b NoMem: MXML couldn't allocate enough memory to complete the operation.
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
   - @b Hierarchy: Broken hierarchy; given node is not in a valid tree.
   - @b CommentInvalid: The comment node is not correctly closed by a --> sequence.
   - @b MultipleXmlDecl: the PI ?xml is declared more than once, or after another node.
*/

ClassErrorCode::ClassErrorCode():
         Enum("ErrorCode")
{
   addConstant("None", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errNone );
   addConstant("IO", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errIo );
   addConstant("NoMem", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errNomem );
   addConstant("InvalidNode", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errInvalidNode );
   addConstant("InvalidAtt", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errInvalidAtt );
   addConstant("MalformedAtt", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errMalformedAtt );
   addConstant("InvalidChar", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errInvalidChar );
   addConstant("Unclosed", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errUnclosed );
   addConstant("UnclosedEntity", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errUnclosedEntity );
   addConstant("WrongEntity", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errWrongEntity );
   addConstant("ChildNotFound", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errChildNotFound );
   addConstant("AttrNotFound", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errAttrNotFound );
   addConstant("Hierarchy", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errHierarchy );
   addConstant("CommentInvalid", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errCommentInvalid );
   addConstant("MultipleXmlDecl", FALCON_MXML_ERROR_BASE + (int) MXML::Error::errMultipleXmlDecl );
}

ClassErrorCode::~ClassErrorCode()
{}

//=======================================================================================
// The XML Module
//=======================================================================================


/*#
   @module mxml  Minimal XML support.
   @ingroup feathers
   @brief Minimal XML support.

   The @b mxml module is a very simple, fast and powerful XML parser
   and generator. It's not designed to be DOM compliant;
   W3C DOM compliance requires some constraints that slows down
   the implementation and burden the interface.

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
      to define and maintain namespaces at application level.

   Apart from this limitations, the support for XML files is complete and
   features as advanced search patterns, node retrieval through XML paths, and
   comment node management are provided.

   To access the functionalities of this module, load it with the instruction
   @code
      import from mxml in mxml
   @endcode
*/

ModuleMXML::ModuleMXML():
         Module(FALCON_FEATHER_MXML_NAME, true)
{
   m_clsNode =  new ClassNode;
   m_clsDoc = new ClassDocument;

   addMantra( m_clsDoc );
   addMantra( m_clsNode );

   addMantra( new ClassMXMLError );
   addMantra( new ClassErrorCode );
   addMantra( new ClassNodeType );
   addMantra( new ClassStyle );
}

ModuleMXML::~ModuleMXML()
{
}

}
}

/* end of mxml_fm.cpp */
