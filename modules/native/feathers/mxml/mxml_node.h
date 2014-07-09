/*
   Mini XML lib PLUS for C++

   Node class

   Author: Giancarlo Niccolai <gian@niccolai.ws>

*/

#ifndef MXML_NODE_H
#define MXML_NODE_H

#include <falcon/string.h>
#include <falcon/stream.h>
#include <list>
#include <cassert>

#include "mxml_error.h"
#include "mxml_attribute.h"

namespace MXML
{

class Document;

typedef std::list<Attribute *>   AttribList;
typedef AttribList::iterator AttribListIter;


template <class __Node>
class __iterator
{
protected:
   __Node *m_base;
   __Node *m_node;
   friend class Node;

   //__iterator( __iterator &src ) { copy( src ); }
   void copy( __iterator &src ) { m_base = src.m_base; m_node = src.m_node; }

   virtual __iterator &__next() {
      assert ( m_node != 0 );
      m_node = m_node->next();
      return *this;
   }
   virtual __iterator &__prev();
public:
   __iterator( __Node *nd=0 ) { m_base = m_node = nd; }
   virtual ~__iterator() {}
   inline __iterator &operator=( __iterator src ) {
      copy( src );
      return *this;
   }
   inline __Node &operator*() const { return *m_node; }
   inline __Node *operator->() const { return &(operator*()); }

   virtual inline __iterator &operator++(){ return __next(); }
   virtual inline  __iterator &operator--() { return __prev(); }

   virtual inline __iterator operator++(int) {
      __iterator tmp = *this;
      operator++();
      return tmp;
   }

   virtual inline __iterator operator--(int) {
      __iterator tmp = *this;
      operator++();
      return tmp;
   }

   inline bool operator==(const __iterator &conf) const
      { return m_node == conf.m_node; }
   inline bool operator!=( const __iterator &conf)
      { return m_node != conf.m_node; }
   inline __iterator &operator+=(const int count);
   inline __iterator &operator-=(const int count);
   inline __iterator operator+(const int count);
   inline __iterator operator-(const int count);
   inline __iterator operator[](const int count) { return operator+(count); }

};

template <class __Node>
class __deep_iterator:public __iterator< __Node>
{
protected:
   friend class Node;

   virtual inline __iterator<__Node> &__next();
   virtual inline __iterator<__Node> &__prev();

public:
   __deep_iterator( __Node *nd=0 ):__iterator< __Node>( nd ) {};
   virtual ~__deep_iterator(){};
};

template <class __Node>
class __find_iterator:public __deep_iterator< __Node>
{
   Falcon::String m_name;
   Falcon::String m_attr;
   Falcon::String m_valattr;
   Falcon::String m_data;
   int m_maxmatch;

protected:
   friend class Node;
   __find_iterator( __Node *nd, const Falcon::String &name, const Falcon::String &attr,
               const Falcon::String &valatt, const Falcon::String &data);

   virtual inline __iterator<__Node> &__next();
   virtual inline __iterator<__Node> &__prev();
   virtual inline __iterator<__Node> &__find();

public:
   __find_iterator( __Node *nd=0 );
   virtual ~__find_iterator() {};

};

template <class __Node>
class __path_iterator:public __iterator< __Node>
{
   Falcon::String m_path;
   virtual inline __Node *subfind( __Node *parent, Falcon::uint32 begin );

protected:
   friend class Node;
   __path_iterator( __Node *nd, const Falcon::String &path );

   virtual inline __iterator<__Node> &__next();
   virtual inline __iterator<__Node> &__prev();
   virtual inline __iterator<__Node> &__find();
public:
   __path_iterator( __Node *nd=0 );
   virtual ~__path_iterator() {};
};

/** Implements an XML node.

   \todo Write serialization progress hooks
*/
class Node: public Element
{

public:

   /** Node types.
      This enum describes the kind of node that is held inside
      this object.
   */
   enum type
   {
      /** The node is a normal XML tag.
         It has possibly attributes, data and surely a name and a path. */
      typeTag=0,
      /** The XML document declaration.
         This node store the content of the <?xml abc ... ?> decl in its name.
         Data is unused, and the node cannot have children.
      */
      typeXMLDecl,
      /** A Comment.
         Comments nodes has only data, representing the content of the comment,
         and has no name, path or attributes. The node cannot have children.
      */
      typeComment,
      /** A CDATA element.
         This is an element that is not to be parsed by XML parsers. Otherwise illegal character sequences,
         as "<", "&something;" and so on are allowed inside CDATA blocks.
         MXML sees a CDATA block as a different block, unless an explicit CDATA flattening operation is
         requrired.
      */
      typeCDATA,
      /** An XML Processing instruction.
         The tag is a processing instruction in the form <?NAME ...?>; all the content
         of the tag, except the name, is set in the data. The node has never children.
      */
      typePI,
      /** An XML directive.
         The directive in the form <!xxx ... > has a name and a content.
         The node has never children.
      */
      typeDirective,
      /** A data node.
         Tag nodes having more than one tag and data nodes below them may have the
         first content in their data, and rely on this kind of node for the other
         data slices right below them. Data nodes have only data; no name, path
         or attribute is meaningful for them.
      */
      typeData,
      /** The root element of a XML document.
         A root node is always empty, and it's kust holding the root level XML nodes.
         Apart from the root tag, it has the XML as child, and may have many comments
         and directives as well.
      */
      typeDocument,   // used for document level root node
      /** Temporary node used internally during deserialization. */
      typeFakeClosing // fake node type for </name> closing tags
   };

private:
   type m_type;
   bool m_bReserve;
   Falcon::String m_name;
   Falcon::String m_data;
   AttribList m_attrib;
   AttribList::iterator m_lastFound;

   Node *m_parent;
   Node *m_child;
   Node *m_last_child;
   Node *m_next;
   Node *m_prev;

   Document* m_doc;
   Falcon::uint32 m_mark;

protected:
   void nodeIndent( Falcon::TextWriter &out, const int depth, const int style ) const;
   void readData( Falcon::TextReader &in, const int iStyle );

   void setDownMark( Falcon::uint32 m );
public:

   /* Creates a new node
      Depending on the types the node could have a name, a data or both.
      \todo chech for name validity and throw an error
      @param tp one of the MXML::Node::type enum
      @param name the name of the newborn node
      @param type the value of the newborn attribute
   */
   Node( const type tp=typeTag, const Falcon::String &name = "", const Falcon::String &data = "" );

   /** Deserializes a node
      Reads a node from an XML file at current position.

      In case of error Throws an MXML::IOError or MXML::MalformedError.

      @param in the input stream
      @param style style bits; see MXML::Document::setStyle()
      @param line the current line in the stream
      @param pos the current position in line
      @throws MXML::MalformedError if the node is invalid
      @throws MXML::IOError in case of hard errors on the stream
   */
   void read( Falcon::TextReader &in, const int style = 0, const int line=1, const int pos=0  )
      throw( MalformedError );

   /** Copy constructor.
      See clone()
   */
   Node( Node & );

   /** Deletes the node.
      All the attributes in the attribute list are deleted also, as well
      as the children of the node. To avoid deleting the children do an
      unlinkComplete of the parent after having got the first child;
      then insert it below or after another node, or iterate through an
      addBelow():

      \code
         // saving children nodes
         Node *child = dying->child();
         dying->unlinkComplete();

         //alternative 1:
         safe->insertBelow( child );

         //alternative 2:
         Node *ch = child;
         while( ch != 0 ) {
            ch = child->next();
            safe->addBelow( child );
            child = ch; // addBelow destroys child->next;
         }

         // do the job
         delete dying;
      \endcode
   */
   ~Node();

   /** Returns the type of this node */
   type nodeType() const { return m_type; }

   void nodeType( type t ) { m_type = t; }

   /** Returns current name of the node.
      If the name is not defined, it returns an empty string.
   */
   const Falcon::String &name() const { return m_name; }
   /** Returns the data element of the node.
      If the name is not defined, it returns an empty string.
   */
   const Falcon::String &data() const { return m_data; }

   /** Change name of the node.
      If the node should not have a name (i.e. comments) this value will
      be ignored by the system.
      \todo check validity of the name and throw a malformed error if wrong.
   */
   void name( const Falcon::String &new_name ) { m_name = new_name; }

   /** Change data of the node.
      The user can also set the data for a node that should not have it (i.e.
      an XMLdecl type), but the data will be ignored by the system.
      \todo check validity of the name and throw a malformed error if wrong.
   */
   void data( const Falcon::String &new_value ) { m_data = new_value; }

   /** Adds a new attribute at the end of the attribute list.
   */
   void addAttribute( Attribute *attrib )
   {
      m_attrib.push_back( attrib );
   }

   /** Gets the value of the given attribute.
      If the attribute is present, its value is returned. If it is
      not present, an MXML::NotFoundError error is thrown.
      @param name the attribute name
      @return the attribute value, if it exists
      @throws MXML::NotFoundError if the attribute name can't be found.
   */
   const Falcon::String getAttribute( const Falcon::String &name ) const
      throw( NotFoundError );

   /** Sets the value of a given attribute.
      This is a shortcut instead of searching for an attribute in
      the list returned by attributes() method and setting its value.
      @param name the attribute name
      @return the attribute value, if it exists
      @throws MXML::NotFoundError if the attribute name can't be found.
   */
   void setAttribute( const Falcon::String &name, const Falcon::String &value )
      throw( NotFoundError );

   /** Returns true if the node has a given attribute.
      The found attribute is cached, so if you make this method follow
      by a getAttribute, the whole operation is done efficiently:
      \code
         ...
         if ( node.hasAttribute( "position" ) )
            pos = node.getAttribute( "position");
         else {
            pos = "0";
            node.addAttribute( new Attribute( "position", pos ) );
         }
         ...
      \endcode
      @param name attribute to be found
      @return true if attribute with given name has been found, false otherwise
   */
   bool hasAttribute( const Falcon::String &name ) const;

   /** Detaches current node from its parent and brothers.
      The node still retain its children though.
   */
   void unlink();

   /** Detaches current node from all the adiacents node.
      Be sure to have a reference to the children nodes, or the whole
      children hierarcy will be lost (causing a memory leak).
      The ideal is to use a Node list to save the old children; the list
      can then be added to other nodes children, like that:
      \code
         ...
         Node *kids;
         kids = myNode->unlinkComplete();

         targetNode.insertBelow( kids );
         ...
      \endcode
      @return a pointer to the first list where the children are saved.
   */
   Node * unlinkComplete();

   /** Removes a child from the children list.
      The child is not deleted; an error is thrown if the parameter
      is not a child of this object.
      \note is possible to check if the node is in the child
         list of this object by checking that its parent() points
         to this. This prevents uslesess try/catches.

      @param child the node to be removed
      @throw NotFoundError if the node is not in the child list
   */
   void removeChild( Node *child ) throw( NotFoundError );


   Node *parent() const { return m_parent; }
   Node *child() const { return m_child; }
   Node *next() const { return m_next; }
   Node *prev() const { return m_prev; }
   Node *lastChild() const { return m_last_child; }


   void addBelow( Node * );
   void insertBelow( Node * );
   void insertBefore( Node * );
   void insertAfter( Node * );

   /** Returns the depth of the path leading to this node.
      ...Or how many levels are above this node.
      Can be called also for nodes that have not a valid symbolic path.
   */
   int depth() const;

   /** Returns a symbolic path leading to the node.
      A Node path is the the list of all the ancestor node names, plus the
      name of the current node, separated by a "/" sign. So, if you
      have a doucument like:
      \code
         <root>
            <item>an item</item>
         </root>
      \endcode

      the path to item will "/root/item".

      Node paths have not to be unique (two nodes can have the same name
      and be brothers). A node can also have an empty path if it has not
      a name, or if it is a child of nodes that have not a name. I.e.
      data nodes and comments have not a path.
      @return the path leading to the node, or an empty string if the node
         has not a valid path.
   */
   Falcon::String path() const;


   /* Clones the node and all its children.
      This effectively creates a copy of the tree of this object.
      @return the new node already linked to its children.
   */
   Node *clone();

   /** Writes the node on a stream.
      This function is usually called by MXML::Document::write(), or by
      the << operator (in class MXML::Element), use the latter only
      for debug.
      @param stream the stream where the object will be written
      @param style the style of the serialization
   */
   virtual void write( Falcon::TextWriter &out, const int style ) const;


   typedef __iterator<Node> iterator;
   typedef __iterator<const Node> const_iterator;
   typedef __deep_iterator<Node> deep_iterator;
   typedef __deep_iterator<const Node> const_deep_iterator;
   typedef __find_iterator<Node> find_iterator;
   typedef __find_iterator<const Node> const_find_iterator;
   typedef __path_iterator<Node> path_iterator;
   typedef __path_iterator<const Node> const_path_iterator;

   iterator begin() { return  m_child; }
   iterator end() { return static_cast<Node *>(0); }
   const_iterator const_begin() const { return static_cast<const Node *>(m_child); }
   const_iterator const_end() { return static_cast<const Node *>(0); }
   deep_iterator deep_begin() { return  m_child; }

   /** Find one or more with specific characteristics.
      This function returns an iterator that iterates over all the nodes below this
      one that matches the given criteria:
      -# Having node name equal to the \b name parameter.
      -# Having an attribute named as \b attrib parameter
      -# \b attrib attribute having value \b valattr
      -# \b data being a substring of node data.

      In case one or more of the elements is to be ignored, it just set the corresponding
      parameter to an empty string (""); if all the parameters are set to an empty
      string, the iterator will iterate over all the nodes below this one.

      The search is done deeply: all the nodes below this one that matches the search criteria
      are returned, regardless of their level and distance from this node.

      In example, to search for all the nodes named "mynode" having the word "good" in their
      data, perform the following scan:

      \code
         MXML::find_iterator iter = currentNode->find( "mynode", "", "", "good" );
         while( iter != currentNode->end() )
         {
            std::out << "Data inside node MYNODE: " << endl;
            std::cout << (*iter)->data() << endl;
            iter++;
         }
      \endcode

      \todo add a flat find iterator.

      If the find is unsuccesful, MXML::Node::end() will be returned.

      \param name Node name to search for or "" to ignore this parameter.
      \param attrib attribute that the searched node must possess, or "" for none.
      \param valatt value that attribute attrib must possess, or any value in the
         attribute list if attrib is ""; set to "" to ignore.
      \param data a substring that must appare in the node data; set to "" to ignore.
      \return an iterator that will return all the nodes that match the critria in
         turn or end().
   */

   find_iterator find( const Falcon::String &name,
                       const Falcon::String &attrib="",
                       const Falcon::String &valatt="",
                       const Falcon::String &data="" );

   /** Recursive node path find iterator.
      \note currently, it only supports complete path or path aliased with '*'; also, it supports only
         the FIRST branch that corresponds to the path and all its leaves.
    */
   path_iterator find_path( const Falcon::String &path );

   /** Falcon extension */
   const AttribList &attribs() const { return m_attrib; }

   void gcMark( Falcon::uint32 m );
   Falcon::uint32 currentMark() const { return m_mark; }
   // delete all the nodes from the topmost.
   void dispose();

   friend class Document;
};

#include "mxml_iterator.h"

} //namespace


#endif
/* end of mxml_node.cpp */
