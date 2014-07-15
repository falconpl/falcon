/*
   Mini XML lib PLUS for C++

   Node class

   Author: Giancarlo Niccolai <gian@niccolai.ws>

*/

#include "mxml.h"
#include "mxml_error.h"
#include "mxml_node.h"
#include "mxml_utility.h"

#include <ctype.h>

#define STATUS_DONE              -1
#define STATUS_BEGIN             0
#define STATUS_FIRSTCHAR         1
#define STATUS_MAYBE_COMMENT     2
#define STATUS_MAYBE_COMMENT2    3
#define STATUS_MAYBE_CDATA0      20
#define STATUS_MAYBE_CDATA1      21
#define STATUS_MAYBE_CDATA2      22
#define STATUS_MAYBE_CDATA3      23
#define STATUS_MAYBE_CDATA4      24
#define STATUS_MAYBE_CDATA5      25
#define STATUS_READ_CDATA        26
#define STATUS_READ_CDATA_END0   27
#define STATUS_READ_CDATA_END1   28

#define STATUS_READ_COMMENT      30
#define STATUS_END_COMMENT1      31
#define STATUS_END_COMMENT2      32
#define STATUS_READ_DATA         40
#define STATUS_READ_ENTITY       41
#define STATUS_READ_TAG_NAME     50
#define STATUS_READ_TAG_NAME2    51
#define STATUS_READ_TAG_NAME3    52
#define STATUS_READ_ATTRIB       60
#define STATUS_READ_SUBNODES     70



namespace MXML
{

Node::Node( const type tp, const Falcon::String &name, const Falcon::String &data ):
   Element()
{
   m_type = tp;
   m_name = name;
   m_data = data;
   m_lastFound = m_attrib.end();

   m_child = m_last_child = m_prev = m_next = m_parent = 0;
   m_mark = 0;
   m_doc = 0;
}


void Node::read( Falcon::TextReader &in, const int style, const int l, const int pos )
   throw( MalformedError )
{
   setPosition( l, pos );
   // variables to optimize data node promotion in tag nodes
   bool promote_data = true;
   Node *the_data_node = 0;
   Falcon::uint32 chr;
   Falcon::String entity;
   int iStatus = STATUS_BEGIN;

   m_prev = m_next = m_parent = m_child = m_last_child = 0;
   // defaults to data type: parents will ignore/destroy empty data elements
   m_type = typeData;

   while ( iStatus >= 0 && ! in.eof() )
   {
      // resetting new node foundings
      chr = nextChar(in);

      //std::cout << "CHR: " << chr << " - status: " << iStatus << std::endl;

      switch ( iStatus ) {

         case STATUS_BEGIN:  // outside nodes
            switch ( chr ) {
               // We repeat line terminator here for portability
               case MXML_SOFT_LINE_TERMINATOR: break;
               case ' ': case '\t': break;
               case '<': iStatus = STATUS_FIRSTCHAR; break;
               case '&': iStatus = STATUS_READ_ENTITY; break;
               default:  // it is a data node
                  m_type = typeData;
                  m_data = chr;
                  iStatus = STATUS_READ_DATA; // data
                  break;
            }
         break;

         case STATUS_FIRSTCHAR: //inside a node, first character
            if ( chr == '/' ) {
               iStatus = STATUS_READ_TAG_NAME;
               m_type = typeFakeClosing;
            }
            else if ( chr == '!' ) {
               iStatus = STATUS_MAYBE_COMMENT;
            }
            else if ( chr == '?' ) {
               m_type = typePI;
               iStatus = STATUS_READ_TAG_NAME; // PI - read node name
            }
            else if ( isalpha( chr ) ) {
               m_type = typeTag;
               m_name = chr;
               iStatus = STATUS_READ_TAG_NAME2; // tag - read node name (2nd char)
            }
            else {
               throw MalformedError( Error::errInvalidNode, this );
            }
         break;

         case STATUS_MAYBE_COMMENT: //inside a possible comment (<!-/<!?)
            if ( chr == '-') {
               iStatus = STATUS_MAYBE_COMMENT2 ;
            }
            else if ( isalpha( chr ) ) {
               m_type = typeDirective;
               m_name = chr;
               iStatus = STATUS_READ_TAG_NAME; // read directive
            }
            else if ( chr == '[' )
            {
               iStatus = STATUS_MAYBE_CDATA0;
            }
            else {
               throw MalformedError( Error::errInvalidNode, this );
            }
         break;

         case STATUS_MAYBE_COMMENT2:
            if ( chr == '-') {
               m_type = typeComment;
               iStatus = STATUS_READ_COMMENT; // read comment
            }
            else {
               throw MalformedError( Error::errInvalidNode, this );
            }
         break;

         case STATUS_MAYBE_CDATA0:
            if ( chr == 'C') {
               iStatus = STATUS_MAYBE_CDATA1; // we have "<!["
            }
            else {
               throw MalformedError( Error::errInvalidNode, this );
            }
         break;

         case STATUS_MAYBE_CDATA1:
            if ( chr == 'D') {
               iStatus = STATUS_MAYBE_CDATA2; // we have "<![C"
            }
            else {
               throw MalformedError( Error::errInvalidNode, this );
            }
         break;

         case STATUS_MAYBE_CDATA2:
            if ( chr == 'A') {
               iStatus = STATUS_MAYBE_CDATA3; // we have "<![CD"
            }
            else {
               throw MalformedError( Error::errInvalidNode, this );
            }
         break;

         case STATUS_MAYBE_CDATA3:
            if ( chr == 'T') {
               iStatus = STATUS_MAYBE_CDATA4; // we have "<![CDA"
            }
            else {
               throw MalformedError( Error::errInvalidNode, this );
            }
         break;

         case STATUS_MAYBE_CDATA4:
            if ( chr == 'A') {
               iStatus = STATUS_MAYBE_CDATA5; // we have "<![CDAT"
            }
            else {
               throw MalformedError( Error::errInvalidNode, this );
            }
         break;

         case STATUS_MAYBE_CDATA5:
            if ( chr == '[') {
               iStatus = STATUS_READ_CDATA; // we have "<![CDATA" +  the final "["
               m_type = typeCDATA;
            }
            else {
               throw MalformedError( Error::errInvalidNode, this );
            }
         break;

         case STATUS_READ_CDATA:
            if ( chr == ']') {
               iStatus = STATUS_READ_CDATA_END0; // "]"
            }
            else {
               m_data += chr;
            }
         break;

         case STATUS_READ_CDATA_END0:
            if ( chr == ']') {
               iStatus = STATUS_READ_CDATA_END1; // "]]"
            }
            else {
               m_data += ']';
               m_data += chr;
               iStatus = STATUS_READ_CDATA;
            }
         break;

         case STATUS_READ_CDATA_END1:
            if ( chr == '>') {
               iStatus = STATUS_DONE;; // "]]>"
            }
            else {
               m_data += ']';
               m_data += ']';
               m_data += chr;
               iStatus = STATUS_READ_CDATA;
            }
         break;

         case STATUS_READ_COMMENT:
            if ( chr == '-' ) {
               iStatus = STATUS_END_COMMENT1;
            }
            else {
               m_data += chr;
            }
         break;

         case STATUS_END_COMMENT1:
            if( chr == '-' )
               iStatus =  STATUS_END_COMMENT2;
            else {
               iStatus = STATUS_READ_COMMENT;
               m_data += "-";
               m_data += chr;
            }
         break;

         case STATUS_END_COMMENT2:
            if ( chr == '>' ) {
               // comment is done!
               iStatus = STATUS_DONE;
            }
            else // any sequence of -- followed by any character != '>' is illegal
               throw MalformedError( Error::errCommentInvalid, this );
         break;

         // data:
         case STATUS_READ_DATA:
            if ( chr == '?' && ( m_type == typePI || m_type == typeXMLDecl ))
               iStatus = STATUS_READ_TAG_NAME3;
            else if ( chr == '>' && m_type != typeData ) {
               // done with this node (either PI or Directive)
               iStatus = STATUS_DONE;
            }
            else if ( chr == '<' && m_type == typeData ) {
               // done with data elements
               ungetChar( in, chr );
               iStatus = STATUS_DONE;
            }
            else {
               if ( m_type == typeData && chr == '&' &&
                     ! ( style & MXML_STYLE_NOESCAPE) )
               {
                  iStatus = STATUS_READ_ENTITY;
                  entity = "";
               }
               else{
                  m_data += chr;
               }
            }
         break;

         // data + escape
         case STATUS_READ_ENTITY:
            if ( chr == ';' ) {
               // we see if we have a predef entity (also known as escape)
               //std::cout << "ENTYTY: " << entity << std::endl;
               if ( ( chr = parseEntity( entity ) ) != 0 )
                  m_data += chr;
               else
                  m_data = m_data + "&" + entity + ";";

               iStatus = STATUS_READ_DATA;
            }
            else if ( !isalnum( chr ) && chr != '_' && chr != '-' && chr != '#' )
               //error - we have something like &amp &amp
               throw MalformedError( Error::errUnclosedEntity, this );
            else
               entity += chr;
         break;

         //Node name, first character
         case STATUS_READ_TAG_NAME:
            if ( isalpha( chr ) ) {
               m_name += chr;
               iStatus = STATUS_READ_TAG_NAME2; // second letter on
            }
            else
               throw MalformedError( Error::errInvalidNode, this );
         break;

         //Node name, from second character on
         case STATUS_READ_TAG_NAME2:
            if ( isalnum( chr ) || chr == '-'
                  || chr == '_' || chr == ':')
               m_name += chr;
            else if ( chr == '/' && m_type != typeFakeClosing )
               iStatus = STATUS_READ_TAG_NAME3; // waiting for '>' to close the tag
            else if ( chr == '?' && ( m_type == typePI || m_type == typeXMLDecl ))
               iStatus = STATUS_READ_TAG_NAME3;
            else if ( chr == '>') {
               if ( m_type == typeFakeClosing )
                  iStatus = STATUS_DONE;
               else
                  iStatus = STATUS_READ_SUBNODES;  // reading subnodes
            }
            else if ( chr == ' ' || chr == '\t' || chr == MXML_SOFT_LINE_TERMINATOR ||
                  chr == MXML_LINE_TERMINATOR )
            {
               // check for xml PI.
               if ( m_type == typePI && m_name == "xml" ) {
                  m_type = typeXMLDecl;
               }

               if ( m_type == typeTag || m_type == typeXMLDecl )
                  iStatus = STATUS_READ_ATTRIB; // read attributes
               else
                  iStatus = STATUS_READ_DATA; // read data.

            }
            else
               throw MalformedError( Error::errInvalidNode, this );
         break;

         // node name; waiting for '>'
         case STATUS_READ_TAG_NAME3:
            if ( chr != '>' )
               throw MalformedError( Error::errInvalidNode, this );
            // if not, we are done with this node
         return;

         // reading attributes
         case STATUS_READ_ATTRIB:
            if ( chr == '/' ||
                  ( chr == '?' && ( m_type == typePI || m_type == typeXMLDecl )))
            {
               iStatus = STATUS_READ_TAG_NAME3; // node name, waiting for '>'
            }
            else if ( chr == '>' )
               iStatus = STATUS_READ_SUBNODES; // subnodes
            else if ( chr == MXML_LINE_TERMINATOR || chr == ' ' || chr == '\t'
                        || chr == MXML_SOFT_LINE_TERMINATOR )
            {
            }
            else {
               ungetChar( in, chr );
               setPosition( line(), character() -1 );
               Attribute *attrib = new Attribute( in, style, line(), character() );
               m_attrib.push_back( attrib );
               setPosition( attrib->line(), attrib->character() );
            }
         break;

         case STATUS_READ_SUBNODES:
            ungetChar( in, chr );
            while ( ! in.eof() ) {
               //std::cout << "Reading subnode" << std::endl;
               Node *child = new Node();
               try {
                  child->read( in, style, line(), character() -1);
               }
               catch( MalformedError &err )
               {
                  delete child;
                  throw err;
               }

               setPosition( child->line(), child->character() );

               if ( child->m_type == typeData )
               {
                  if ( child->m_data == "" )  // delete empty data nodes
                     delete child;
                  else {
                     // set the-data-node for data promotion
                     if ( the_data_node == 0 )
                        the_data_node = child;
                     else
                        promote_data = false;
                     addBelow( child );
                  }
               }
               // have we found our closing node?
               else if ( child->m_type == typeFakeClosing ) {
                  //is the name valid?
                  if ( m_name == child->m_name ) {
                     iStatus = STATUS_DONE;
                     delete child;
                     break;
                  }
                  else { // We are unclosed!
                     delete child;
                     throw MalformedError( Error::errUnclosed, this );
                  }
               }
               else // in all the other cases, add subnodes.
                  addBelow( child );

            }
         break;
      } // switch

   } // while

   // now we do a little cleanup:
   // if we are a data or a comment node, trim the data
   // if we are a tag and we have just one data node, let's move it to our
   //   data member
   if ( m_type == typeData || m_type == typeComment )
   {
      m_data.trim();
   }

   if ( m_type == typeTag && promote_data && the_data_node != 0 )
   {
      // only if we have a single child node: the data child.
      if ( child() == the_data_node && lastChild() == the_data_node )
      {
         m_data = the_data_node->m_data;
         // Data node have not children, and delete calls unlink()
         delete the_data_node;
      }
   }

}

/************************************************/

Node::Node( Node &src ) :
   Element( src )
{
   m_type = src.m_type;
   m_name = src.m_name;
   m_data = src.m_data;
   m_bReserve = false;

   AttribList::iterator iter = src.m_attrib.begin();

   while( iter != src.m_attrib.end() ) {
      m_attrib.push_back( new Attribute( *(*iter ) ) );
      iter ++;
   }

   iter = m_attrib.end();

   m_parent = m_child = m_last_child = m_next = m_prev = 0;
   m_mark = 0;
   m_doc = 0;
}

Node::~Node()
{
   unlink();

   AttribList::iterator iter = m_attrib.begin();
   while( iter != m_attrib.end() ) {
      delete (*iter);
      iter ++;
   }

   Node *child = m_child;
   while( child != 0 ) {
      Node *tmp = child;
      child = child->m_next;
      delete tmp;
   }
}

/* Search routines */

const Falcon::String Node::getAttribute( const Falcon::String &name ) const
   throw( NotFoundError )
{
   AttribList::const_iterator iter = m_attrib.begin();

   while ( iter != m_attrib.end() ) {
      if ( (*iter)->name() == name )
         return (*iter)->value();
      iter++;
   }
   throw NotFoundError( Error::errAttrNotFound, this );
}

void Node::setAttribute( const Falcon::String &name, const Falcon::String &value )
   throw( NotFoundError )
{
   AttribList::iterator iter = m_attrib.begin();

   while ( iter != m_attrib.end() ) {
      if ( (*iter)->name() == name ) {
         (*iter)->value( value );
         return;
      }
      iter++;
   }
   throw NotFoundError( Error::errAttrNotFound, this );
}

bool Node::hasAttribute( const Falcon::String &name ) const
{
   AttribList::const_iterator iter = m_attrib.begin();

   while ( iter != m_attrib.end() ) {
      if ( (*iter)->name() == name )
         return true;
      iter++;
   }
   return false;
}

/* unlink routines */
void Node::removeChild( Node *child )
   throw( NotFoundError )
{
   if (child->parent() != this )
      throw NotFoundError( Error::errHierarchy, this );

   if ( m_child == child )
      m_child = child->m_next;
   if ( m_last_child == child )
      m_last_child = child->m_prev;

   if ( child->m_next != 0 )
      child->m_next->m_prev = child->m_prev;

   if ( child->m_prev != 0 )
      child->m_prev->m_next = child->m_next;

   child->m_parent = 0;
   child->m_next = 0;
   child->m_prev = 0;
}


void Node::unlink()
{
   if ( m_parent != 0 ) {
      m_parent->removeChild( this );
      m_parent = 0;
   }
   else {
      if ( m_next != 0 )
         m_next->m_prev = m_prev;

      if ( m_prev != 0 )
         m_prev->m_next = m_next;
   }
}

Node *Node::unlinkComplete()
{
   unlink();

   Node *child = m_child, *old = m_child;

   while ( child != 0 ) {
      child->m_parent = 0;
      child = child->m_next;
   }

   m_child = 0;
   m_last_child = 0;

   return old;
}

/* modify routines */
void Node::addBelow( Node *node )
{
   if ( node->m_parent == this ) return;
   if ( node->m_parent != 0 )
      node->m_parent->removeChild( node );

   node->m_parent = this;
   node->m_next = 0;  // just a precaution

   if ( m_last_child != 0 ) {
      m_last_child->m_next = node;
      node->m_prev = m_last_child;
      m_last_child = node;
   }
   else {
      node->m_prev = 0;
      m_last_child = m_child = node;
   }
}

void Node::insertBelow( Node *node )
{
   if ( node->m_parent == this ) return;
   if ( node->m_parent != 0 )
      node->m_parent->removeChild( node );

   node->m_parent = this;
   node->m_prev = 0;
   node->m_next = m_child;
   if ( m_child != 0 )
      m_child->m_prev = node;
   m_child = node;
}

void Node::insertBefore( Node *node )
{
   node->m_next = this;
   node->m_prev = m_prev;
   node->m_parent = m_parent;
   /* puts the node on top of hierarchy if needed */
   if ( m_parent != 0 && m_parent->m_child == this )
      m_parent->m_child = node;

   m_prev = node;
}

void Node::insertAfter( Node *node )
{
   node->m_next = m_next;
   node->m_prev = this;
   node->m_parent = m_parent;
   /* puts the node on bottom of hierarchy if needed */
   if ( m_parent != NULL && m_parent->m_last_child == this )
      m_parent->m_last_child = node;

   m_next = node;
}


Node *Node::clone()
{
   Node *copy = new Node( *this );
   Node *node = m_child;

   if ( node != 0 ) {
      copy->m_child = node->clone();
      copy->m_child->m_parent = copy;
      copy->m_last_child = copy->m_child;
      node = node->m_next;

      while ( node != 0 ) {
         copy->m_last_child->m_next = node->clone();
         copy->m_last_child->m_next->m_parent = copy;
         copy->m_last_child->m_next->m_prev = copy->m_last_child->m_next;
         copy->m_last_child = copy->m_last_child->m_next;
         node = node->m_next;
      }
   }

   copy->m_doc = 0;
   return copy;
}

int Node::depth() const
{
   int iDepth = 0;
   const Node *parent = this;

   while( parent != 0 && parent->m_type != typeDocument ) {
      iDepth++;
      parent = parent->m_parent;
   }

   return iDepth;
}

Falcon::String Node::path() const
{
   Falcon::String ret = "";
   const Node *parent = this;

   while( parent != 0 && parent->m_name != "") {
      ret = "/" + parent->m_name + ret;
      parent = parent->m_parent;
   }

   return ret;
}

/************************************************
* Find section
************************************************/
Node::find_iterator
   Node::find( const Falcon::String &name, const Falcon::String &attrib, const Falcon::String &valatt, const Falcon::String &data )
{
   find_iterator iter( this, name, attrib, valatt, data);
   return iter;
}

Node::path_iterator Node::find_path( const Falcon::String &path )
{
   path_iterator iter( this, path );
   return iter;
}

/************************************************
* Write sections
************************************************/

void Node::nodeIndent( Falcon::TextWriter &out, const int iDepth, const int style ) const
{
   int i;

   for ( i = 0; i < iDepth; i++ ) {
      if ( style & MXML_STYLE_TAB )
         out.putChar( '\t' );
      else if (  style & MXML_STYLE_THREESPACES )
         out.write( "   " );
      else
         out.putChar( ' ' );
   }

}

void Node::write( Falcon::TextWriter &out, const int style ) const
{
   Node *child;
   int iDepth = 0;
   int mustIndent = 0;
   AttribList::const_iterator iter;

   if ( style & MXML_STYLE_INDENT ) {
      iDepth = depth()-1;
      nodeIndent( out, iDepth, style );
   }

   switch( m_type ) {
      case typeTag:
         out.putChar( '<' );
         out.write( m_name );

         iter = m_attrib.begin();
         while( iter != m_attrib.end() ) {
            out.putChar( ' ' );
            (*iter)->write( out, style ) ;
            iter++;
         }

         if ( m_data == "" && m_child == 0 ) {
            out.write( "/>\n" );
         }
         else {
            out.putChar( '>' );

            child = m_child;
            if ( child != 0 ) {
               mustIndent = 1;
               out.putChar( '\n' );

               while ( child != 0 ) {
                  child->write( out, style );
                  child = child->m_next;
               }
            }

            if ( m_data != "" ) {
               if ( mustIndent && ( style & MXML_STYLE_INDENT ) )
                  nodeIndent( out, iDepth+1, style );

               if ( style & MXML_STYLE_NOESCAPE )
                  out.write( m_data );
               else
                  MXML::writeEscape( out, m_data );

               if ( mustIndent )
               {
                  out.putChar( '\n' );
               }
            }

            if ( mustIndent && ( style & MXML_STYLE_INDENT ))
            {
               nodeIndent( out, iDepth, style );
            }

            out.write( "</" );
            out.write( m_name );
            out.write( ">\n" );
         }
      break;

      case typeComment:
            out.write( "<!-- " );
            out.write( m_data );
            out.write( " -->\n" );
      break;

      case typeData:
         if ( style & MXML_STYLE_NOESCAPE )
            out.write( m_data );
         else
            MXML::writeEscape( out, m_data );
         out.putChar( '\n' );
      break;

      case typeCDATA:
         out.write( "<![CDATA[" );
         out.write( m_data );
         out.write( "]]>\n" );
      break;

      case typeDirective:
         out.write( "<!" );
         out.write( m_name );
         out.putChar( ' ' );
         out.write( m_data );
         out.write( ">\n" );
      break;

      case typePI:
         out.write( "<?" );
         out.write( m_name );
         out.putChar( ' ' );
         out.write( m_data );
         out.write( ">\n" );
      break;

      case typeDocument:
         child = m_child;
         while ( child != 0 ) {
            child->write( out, style );
            child = child->m_next;
         }
         out.putChar( '\n' );
      break;
         
      default:
         break;
   }

}


void Node::setDownMark( Falcon::uint32 m )
{
   if( m_mark != m )
   {
      m_mark = m;
      Node* child = m_child;
      while( child != 0 )
      {
         child->setDownMark(m);
         child = child->m_next;
      }
   }
}


void Node::gcMark( Falcon::uint32 m )
{
   if( m != m_mark )
   {
      if ( m_parent != 0 )
      {
         m_parent->gcMark( m );
      }
      else
      {
         setDownMark( m );
         if( m_doc != 0 )
         {
            m_doc->gcMark(m);
         }
      }
   }
}


void Node::dispose()
{
   if( m_parent != 0 )
   {
      m_parent->dispose();
   }
   else if( m_doc != 0 )
   {
      delete m_doc;
   }
   else {
      delete this;
   }
}

}

/* end of mxml_node.cpp */
