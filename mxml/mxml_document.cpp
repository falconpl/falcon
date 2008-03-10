/*
   Mini XML lib PLUS for C++

   Document class

   Author: Giancarlo Niccolai <gc@falconpl.org>
*/


#include <mxml_document.h>

namespace MXML {

Document::Document( const Falcon::String &encoding, const int style )
   :Element(),
   m_encoding( encoding )
{
   m_style = style;
   m_root = new Node(Node::typeDocument);
}

Document::Document( Document &doc )
{
   m_style = doc.m_style;
   m_root = doc.m_root->clone();
   m_encoding = doc.m_encoding;
}

Document::Document( Falcon::Stream &in, const int style )
   throw( MalformedError )
{
   m_style = style;
   m_root = new Node( Node::typeDocument );
   read( in );
}

Document::~Document()
{
   m_root->dispose();
}

Node *Document::main() const
{
   Node *ret = m_root->lastChild();
   while ( ret != 0 && ret->nodeType() != Node::typeTag )
      ret = ret->prev();
   return ret;
}

void Document::write( Falcon::Stream &stream, const int style ) const
{
   stream.writeString( "<?xml encoding=\"" + m_encoding + "\"?>\n");
   m_root->write( stream, m_style );
}


void Document::read( Falcon::Stream &stream )
   throw( MalformedError )
{
   if ( m_root->child() != 0 ) {
      m_root->dispose();
      m_root = new Node( Node::typeDocument );
   }

   // load the <?xml document declaration

   bool xmlDecl = false;

   while ( stream.good() )
   {
      // ignore parameter style
      Node *child = new Node( stream, m_style, line(), character());
      setPosition( child->line(), child->character() );
      if( child->nodeType() == Node::typeXMLDecl )
      {
         if ( xmlDecl )
         {
            m_root->addBelow( child );
            throw MalformedError( Error::errMultipleXmlDecl, child );
         }
         xmlDecl = true;

         // find the encoding parameter.
         if ( child->hasAttribute( "encoding" ) )
            m_encoding = child->getAttribute( "encoding" );
         else
            m_encoding = "C";

         delete child;
      }

      if ( child->nodeType() == Node::typeData && child->data() == "" )
         delete child;
      else {
         m_root->addBelow( child );
      }
   }

   if ( stream.bad() )
   {
      throw MalformedError( Error::errIo, m_root );
   }
   //todo: validity checks
}

}

/* end of mxml_document.cpp */
