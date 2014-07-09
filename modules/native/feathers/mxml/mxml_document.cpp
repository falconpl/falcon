/*
   Mini XML lib PLUS for C++

   Document class

   Author: Giancarlo Niccolai <gc@falconpl.org>
*/


#include "mxml_document.h"

namespace MXML {

Document::Document( const Falcon::String &encoding, const int style )
   :Element(),
   m_encoding( encoding )
{
   m_style = style;
   m_root = new Node(Node::typeDocument);
   m_root->m_doc = this;
}

Document::Document( const Document &doc ):
   Element(doc)
{
   m_style = doc.m_style;
   m_root = doc.m_root->clone();
   m_root->m_doc = this;
   m_encoding = doc.m_encoding;
}

Document::Document( Falcon::TextReader &in, const int style )
   throw( MalformedError )
{
   m_style = style;
   m_root = new Node( Node::typeDocument );
   m_root->m_doc = this;
   read( in );
}

Document::~Document()
{
   delete m_root;
}

Node *Document::main() const
{
   Node *ret = m_root->lastChild();
   while ( ret != 0 && ret->nodeType() != Node::typeTag )
      ret = ret->prev();
   return ret;
}

void Document::write( Falcon::TextWriter &stream, const int style ) const
{
   stream.write( "<?xml version=\"1.0\" encoding=\"" + m_encoding + "\"?>\n");
   m_root->write( stream, style );
}


void Document::read( Falcon::TextReader &stream )
   throw( MalformedError )
{
   setPosition( 1, 1);
   if ( m_root->child() != 0 ) {
      // drop the old root
      m_root->m_doc = 0;
      m_root = new Node( Node::typeDocument );
   }

   // load the <?xml document declaration

   bool xmlDecl = false;

   while ( ! stream.eof() )
   {
      // ignore parameter style
      Node *child = new Node();
      try
      {
         child->read( stream, m_style, line(), character());
      }
      catch( ... )
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

   //todo: validity checks
}

void Document::gcMark( Falcon::uint32 m )
{
   if ( m_mark != m )
   {
      m_mark = m;
      m_root->gcMark(m);
   }
}

}

/* end of mxml_document.cpp */
