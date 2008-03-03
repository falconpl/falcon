/*
   Mini XML lib PLUS for C++

   Document class

   Author: Giancarlo Niccolai <gc@falconpl.org>
*/


#include <mxml_document.h>

namespace MXML {

Document::Document( const int style )
   :Element()
{
   m_style = style;
   m_root = new Node(Node::typeDocument);
}

Document::Document( Document &doc )
{
   m_style = doc.m_style;
   m_root = doc.m_root->clone();
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
   delete m_root;
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
   // ignore parameter style
   m_root->write( stream, m_style );
}


void Document::read( Falcon::Stream &stream )
   throw( MalformedError )
{
   if ( m_root->child() != 0 ) {
      delete m_root;
      m_root = new Node(Node::typeDocument );
   }

   while ( stream.good() ) {
      // ignore parameter style
      Node *child = new Node( stream, m_style, line(), character());
      setPosition( child->line(), child->character() );
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
