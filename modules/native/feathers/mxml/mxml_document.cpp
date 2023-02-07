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
   m_root->reserve();
}

Document::Document( Document &doc )
{
   m_style = doc.m_style;
   m_root = doc.m_root->clone();
   m_encoding = doc.m_encoding;
}

Document::Document( Falcon::Stream &in, const int style )
{
   m_style = style;
   m_root = new Node( Node::typeDocument );
   m_root->reserve();
   read( in );
}

Document::~Document()
{
   if ( m_root->shell() == 0 )
      delete m_root;
   else
      m_root->unreserve();
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
   stream.writeString( "<?xml version=\"1.0\" encoding=\"" + m_encoding + "\"?>\n");
   m_root->write( stream, m_style );
}


void Document::read( Falcon::Stream &stream )
{
   setPosition( 1, 1);
   if ( m_root->child() != 0 ) {
      m_root->dispose();
      m_root = new Node( Node::typeDocument );
      m_root->reserve();
   }

   // load the <?xml document declaration

   bool xmlDecl = false;

   while ( stream.good() && ! stream.eof() )
   {
      // ignore parameter style
      Node *child = new Node();
      try
      {
         child->read( stream, m_style, line(), character());
      }
      catch( MalformedError )
      {
         delete child;
         throw;
      }

      setPosition( child->line(), child->character() );
      if( child->nodeType() == Node::typeXMLDecl )
      {
         if ( xmlDecl )
         {
            MalformedError err( Error::errMultipleXmlDecl, child );
            delete child;
            throw err;
         }
         xmlDecl = true;

         // find the encoding parameter.
         if ( child->hasAttribute( "encoding" ) )
            m_encoding = child->getAttribute( "encoding" );
         else
            m_encoding = "C";

         delete child;
         continue;
      }

      if ( child->nodeType() == Node::typeData && child->data() == "" )
         delete child;
      else {
         m_root->addBelow( child );
      }
   }

   if ( stream.bad() )
   {
      throw IOError( Error::errIo, m_root );
   }
   //todo: validity checks
}

}

/* end of mxml_document.cpp */
