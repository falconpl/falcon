/*
   FALCON - The Falcon Programming Language.
   FILE: datareader.cpp

   Data-oriented stream reader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Mar 2011 11:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_DATAREADER_H
#define	_FALCON_DATAREADER_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/reader.h>

namespace Falcon {

class Stream;

/** Class providing the ability to read data from a text oriented stream.

    This class is meant to read arbitrary data from a stream. The class
 has support for reading various kind of C/C++ native data, and has an internal
 management of data endianity.

 DataReader and DataWriter classes are used for serialization. As such, they have
 also limited support to read and write basic Falcon data structure (as i.e. strings).

 */
class FALCON_DYN_CLASS DataReader: public Reader
{
public:
   typedef enum {
      e_LE,
      e_BE,
      e_sameEndian,
      e_reverseEndian
   } t_endianity;

   /** Creates a data stream with a predefined endianity.

    Falcon data serialization is Little Endian by default, so DataReader and
    DataWriter have an according endianity default setting.
   */

   DataReader( Stream* stream, t_endianity endian = e_LE, bool bOwn = false );

   /** Constructor for immediate delegation.

    This constructor shall be used when the data reader must receive a stream
    as a delegate from another reader.

    Falcon data serialization is Little Endian by default, so DataReader and
    DataWriter have an according endianity default setting.
   */
   DataReader( t_endianity endian = e_LE );

   virtual ~DataReader();

   /** Sets the endianity of the integer and floating point data reads. */
   void setEndianity( t_endianity endian );

   /** Gets the endianity of this stream. */
   t_endianity endianity() const { return m_endianity; }
   /** Returns true if the stream endianity is the same as the machine endianity */
   bool isSameEndianity() const { return m_bIsSameEndianity; }

   /** Reads a boolean value.
    \return false if the value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
    */
   bool read(bool &value);

   /** Reads a signed 8-bit integer value.
    \return false if the value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
    */
   bool read(char &value);

   /** Reads an unsigned 8-bit integer value.
    \return false if the value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
    */
   bool read(byte &value);

   /** Reads an 16 bit signed integer value.
    \return false if the value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool read(int16 &value);

   /** Reads an 16 bit signed integer value.
    \return false if the value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool read(uint16 &value);

   /** Reads a 32 bit signed integer value.
    \return false if the value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool read(int32 &value);

   /** Reads a 32 bit signed integer value.
    \return false if the value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool read(uint32 &value);


   /** Reads a 64 bit signed integer value.
    \return false if the boolean value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool read(int64 &value);

   /** Reads a 64 bit signed integer value.
    \return false if the value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool read(uint64 &value);


   /** Reads a 32 bit float value.
    \return false if the boolean value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool read(float &value);

   /** Reads a 64 bit float value.
    \return false if the value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool read(double &value);

   /** Reads a long dobule float value.
    \return false if the value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.
   */
   bool read(long double &value);

   /** Reads an arbitrary amount of data.
    \return false if the value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.

    Notice; this differs from the stream read() method; the method may either
    succeed and read the entire size or fail (and eventually throw an error) if
    not enough bytes can be read from the stream.

    \note buffer must be long enough to host size bytes.
    */
   bool read( byte* buffer, length_t size );

   /** Reads a previously written string.
    \return false if the value cannot be read.
    \throw IOError instead of returning false if the underlying stream has throwing
    exceptions enabled -- which is the default for readers.

    This method de-serializes a string previously written by a data writer.

    Falcon Strings are not encoded through a text encoder; they store an internal
    representation of the character values. 
    */
   bool read( String& tgt );

private:

   t_endianity m_endianity;
   bool m_bIsSameEndianity;
};

}

#endif	/* _FALCON_DATAREADER_H */

/* end of datareader.h */
