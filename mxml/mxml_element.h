/*
   Mini XML lib PLUS for C++

   Element class

   Author: Giancarlo Niccolai <gian@niccolai.ws>

   $Id: mxml_element.h,v 1.2 2004/10/14 13:16:37 jonnymind Exp $
*/

#ifndef MXML_ELEMENT_H
#define MXML_ELEMENT_H

#include <falcon/stream.h>
#include <falcon/string.h>

namespace MXML {

/** XML element abstract base class.
This class provides basic functionality for XML parsing; in particular, it
provides a way to keep track of current line/position while parsing the
document, and declares a pure virtual write method (that is called then
by the << operator).
*/

class Element
{
private:
   /** Current processing line in input file */
   int m_line;
   /** Current processing character in current line of input file */
   int m_char;
   /** Line at which the current element begun */
   int m_beginLine;
   /** Character in Line at which the current element begun */
   int m_beginChar;

protected:
   /* Fills current and initial line and character for the current element.
   This constructor can be called by the parent object that is going to
   deserialize a certain element (i.e. a document going to read a node),
   filling it with its own current processing line and character; after the
   deserialization is done, the line and character of the calling process
   should be updated with the line() and characer() method, like this:
   \code
      try {
      ...
         MXML::Node *child = new MXML::Node( in_stream, 0, line(), character() );
         setPosition( child->line(), child->character() );
      }
   \endcode

   @param line current line in file that is being processed
   @param char current character in current line being processed
   */
   Element( const int line=1, const int pos=0 )
   {
      setPosition( line, pos );
      markBegin();
   }

public:
   /** Increments current processing line and set current position in line to 0 */
   void nextLine() { m_line++; m_char = 0; }
   /** Increments current position in line by one. */
   void nextChar() { m_char++; }
   /** Set current position to the given value */
   void setPosition( const int line, const int character ) {
      m_line = line;
      m_char = character;
   }
   /** Returns current line in processing file, or last line processed by this object */
   const int line() const { return m_line; }
   /** Returns current position in line in processing file, or last position processed by this object */
   const int character() const { return m_char; }

   /** Returns the line where this object begun */
   const int beginLine() const { return m_beginLine; }
   /** Returns the position in line where this object begun */
   const int beginChar() const { return m_beginChar; }

   /** Marks current position as the begin of current item.
      The constructor caller will set a default starting line and position, but
      it is possible that i.e. due blanks, the real initial position of the
      current object is later discovered to be elsewhere. Call this method
      to update the current position as the real begin of the object in
      the input stream.
   */
   void markBegin() { m_beginLine = m_line; m_beginChar = m_char; }

   /** Serializes current object to the stream.
      Notice that the << operator won't allow to set style, so using
      stream << element you will not able to set the output style.
      Anyway, the MXML::Document class can set the default style
      for its hyerarcy, and it will take care to pass the given
      style to all the objects, in case you want to serialize it
      with << operator.
      @param stream the stream where the object will be written
      @param style the style of the serialization
      @see MXML::Document::setStyle()
   */
   virtual void write( Falcon::Stream &stream, const int style ) const = 0;

};

} // namespace


#endif
