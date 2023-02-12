/*
   Mini XML lib PLUS for C++

   Document class

   Author: Giancarlo Niccolai <gian@niccolai.ws>

*/

/** \file
   Document class implementation.
   This file contains the implementation of the xml document class.
   \see MXML::Document
*/

#ifndef MXML_DOCUMENT_H
#define MXML_DOCUMENT_H

#include <mxml_node.h>
#include <mxml_element.h>

namespace Falcon {
   class VMachine;
}

namespace MXML {

/** Class containing a logical XML file representation.

   To work with MXML, you need to instantiate at least one object from this class;
   this represents the logic of a document. It is derived for an element, as
   an XML document must still be valid as an XML element of another document,
   but it implements some data specific for handling documents.

   The class has the ability to parse a document into a node tree, deserializing
   it from a std::stream, and to output the document via the >> operator;
   virtual write() and read() functions overloading the element ones are
   also provided.

   The document has an (optional) style that is used on load and eventually on
   write. \see stylemacros
*/

class Document: public Element
{
private:
   Node *m_root;
   int m_style;
   Falcon::String m_encoding;

   // Useful just from find.
   Node::find_iterator m_finditer;
   Node::path_iterator m_pathiter;

public:

   /** Creates the document object.
      This constructor does not load any document, and sets the style parameter to
      nothing (all defaults). Style may be changed later on with the style(int)
      method, but you need to do it before the document si read() if you want to
      set some style affects document parsing.
      \param encoding Uses the given encoding
      \param style the mode in which the document is read/written
      \see stylemacros
      \see read
   */
   Document( const Falcon::String &encoding, const int style = 0 );

   /** Creates a deep copy of the document.
      Each node of the original document is replicated into another separated tree.
   */
   Document( Document &doc );

   /** Creates the document object and reads it.
      The parsing may be unsuccesful, or the reading may be faulty for a
      defective device; if this happens, a MalformedError is thrown.

      \param in the stream in which the data is read.
      \param style the mode in which the document is read/written
      \see stylemacros
   */
   Document( Falcon::Stream &in, const int style = 0 );

   /** Destroys the document.
      If you provided a stream at document creation, the stream is NOT colsed.
      It's up to the caller to destroy the stream and dispose of it cleanly.
   */
   ~Document();

   /** Returns the root node.
      The root node is \e always a root node type; this means that it has no name,
      attributes nor data: it's just a container for the other top level nodes.
   */
   Node *root() const { return m_root; }
   int style() const { return m_style; }
   void style( const int style ) { m_style = style; }

   /** Returns the currently used encoding.
      \note Default encoding is "C"
   */
   const Falcon::String encoding() const { return m_encoding; }

   /** Sets the preferred encoding.
      \note Used to create the main xml node; the actual encoder depends on the stream being used.
      \param enc an encoding ISO code.
   */
   void encoding( const Falcon::String &enc ) { m_encoding = enc; }

   /** Returns the main node, if it exists.
      Finds the main node, that is, the first (and one) tag type node at toplevel.
      If this node does not exists, it returns NULL.
      \return the main node or null
   */
   Node *main() const;

   /** Writes the document.
      \todo throw an exception on error, or return a value.
      Writes the document to the stream; in case of error the
      status of the document will be faulty.
      \see Node::write()
   */
   virtual void write( Falcon::Stream &stream, const int style ) const;

   virtual void write( Falcon::Stream &stream ) const
   {
      write( stream, style() );
   }

   /** Reads and parse the document.
      \todo throw an exception on error, or return a value.
      Reads the document from the stream; in case of error the
      status of the document will be faulty.

      Example:
      \code
      #include <mxml.h>
      #include <iostream>
      #include <fstream>
      ...
      MXML::Document *readXml()
      {
         ...
         ifstream test_file("test.xml");

         if (! test_file.is_open())
         {
            cout << "Error opening file" << endl;
            exit (1);
         }

         MXML::Document *xml_doc = new MXML::Document();
         try {
            xml_doc->read( test_file );
         }
         catch( MXML::Error &er )
         {
            std::cout << std::endl << er << std::endl;
            delete xml_doc;
            xml_doc = 0;
         }
         return xml_doc;
      }
      \endcode

      \param stream the input stream used for reading the data.
      \see Node::read()
   */
   virtual void read( Falcon::Stream &stream );

   /** Falcon MXML extension. */
   Node *find( const Falcon::String &name, const Falcon::String &attr,
               const Falcon::String &va, const Falcon::String &data )
   {
      m_finditer = m_root->find( name, attr, va, data );
      if( m_finditer != m_root->end() )
         return & (*m_finditer);
      return 0;
   }

   /** Falcon MXML extension. */
   Node *findNext() {
      if ( m_finditer == m_root->end() )
         return 0;

      if ( ++m_finditer != m_root->end() )
         return &(*m_finditer);
      return 0;
   }

   /** Falcon MXML extension. */
   Node *findPath( const Falcon::String &path )
   {
      m_pathiter = m_root->find_path( path );
      if( m_pathiter != m_root->end() )
         return & (*m_pathiter);
      return 0;
   }

   /** Falcon MXML extension. */
   Node *findNextPath() {
      if ( m_pathiter == m_root->end() )
         return 0;

      if ( ++m_pathiter != m_root->end() )
         return &(*m_pathiter);
      return 0;
   }
};

class DocumentCarrier: public Falcon::FalconData
{
   Document *m_doc;
public:

   DocumentCarrier( Document *doc ):
      m_doc( doc )
   {
   }

   virtual ~DocumentCarrier()
   {
      delete m_doc;
   }

   virtual void gcMark( Falcon::uint32 mk )  {};
   virtual Falcon::FalconData *clone() const { return 0; }

   Document *document() const { return m_doc; }
};

}
#endif

/* end of mxml_document.cpp */
