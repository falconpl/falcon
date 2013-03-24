/*
   FALCON - The Falcon Programming Language.
   FILE: datawriter.cpp

   Data-oriented stream reader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Mar 2011 21:25:47 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_DATAWRITER_H
#define	_FALCON_DATAWRITER_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/writer.h>

namespace Falcon {

class Stream;
class Date;

/** Class providing the ability to write data to a binary oriented stream.

    This class is meant to write arbitrary data to a stream. The class
 has support for writing various kind of C/C++ native data, and has an internal
 management of data endianity.

 DataReader and DataWriter classes are used for serialization. As such, they have
 also limited support to read and write basic Falcon data structure (as i.e. strings).

 \note DataWriter and DataReader have support to be directly shared with a virtual
 machine by offering a GC hook (gcMarking). A DataReader or DataWriter that might
 be given to the virtual machine should not be directly deleted after isInGC()
 returns true.
 
 \see DataReader
 */
class FALCON_DYN_CLASS DataWriter: public Writer
{
public:
   typedef enum {
      e_LE,
      e_BE,
      e_sameEndian,
      e_reverseEndian
   } t_endianity;

   /** Creates a data stream with a predefined endianity.

    Falcon data serialization is Little Endian by default, so DataWriter and
    DataWriter have an according endianity default setting.
   */

   DataWriter( Stream* stream, t_endianity endian = e_LE );

   /** Constructor for immediate delegation.

    This constructor shall be used when the data reader must receive a stream
    as a delegate from another reader.

    Falcon data serialization is Little Endian by default, so DataWriter and
    DataWriter have an according endianity default setting.
   */
   DataWriter( t_endianity endian = e_LE );
   
   DataWriter( const DataWriter& other );

   virtual ~DataWriter();

   /** Sets the endianity of the integer and floating point data reads. */
   void setEndianity( t_endianity endian );

   /** Gets the endianity of this stream. */
   t_endianity endianity() const { return m_endianity; }

   /** Returns true if the stream endianity is the same as the machine endianity */
   bool isSameEndianity() const { return m_bIsSameEndianity; }

   /** Writes a boolean value.
    \return false if the value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
    */
   bool write(bool value);

   /** Writes a signed 8-bit integer value.
    \return false if the value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
    */
   bool write(char value);

   /** Writes an unsigned 8-bit integer value.
    \return false if the value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
    */
   bool write(byte value);

   /** Writes an 16 bit signed integer value.
    \return false if the value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool write(int16 value) { return write((uint16) value); }

   /** Writes an 16 bit signed integer value.
    \return false if the value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool write(uint16 value);

   /** Writes a 32 bit signed integer value.
    \return false if the value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool write(int32 value) { return write((uint32) value); }

   /** Writes a 32 bit signed integer value.
    \return false if the value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool write(uint32 value);


   /** Writes a 64 bit signed integer value.
    \return false if the boolean value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool write(int64 value) { return write((uint64) value); }

   /** Writes a 64 bit signed integer value.
    \return false if the value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool write(uint64 value);


   /** Writes a 32 bit float value.
    \return false if the boolean value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool write(float value);

   /** Writes a 64 bit float value.
    \return false if the value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool write(double value);

   /** Writes a previously written string.
    \return false if the value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.

    This method writes a string on the target stream.

    When stored this way, Strings are not encoded through a text encoder; they store an internal
    representation of the character values. 
    */
   bool write( const String& tgt );
   
   /** Writes a date on the underlying stream.
    \return false if the value cannot be written.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
    */
   bool write( const Date& date );

   bool write( const char* data ) { return write(String(data)); }

private:

   t_endianity m_endianity;
   bool m_bIsSameEndianity;
};

}

#endif	/* _FALCON_DATAWRITER_H */

/* end of datawriter.h */
