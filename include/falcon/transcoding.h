/*
   FALCON - The Falcon Programming Language.
   FILE: transcoding.h

   Declarations of encoders and decoders.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ago 20 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Declarations of encoders and decoders.
*/

#ifndef flc_transcoding_H
#define flc_transcoding_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/stream.h>

namespace Falcon {

typedef struct tag_cp_iso_uint_table
{
   uint16 unicode;
   uint16 local;
} CP_ISO_UINT_TABLE;

/** Basic transcoder class.
   Falcon strings are internally organized in a format that is meant
   to:
      - store every possible UNICODE character
      - allow scripts to access nth character randomly at cost o(1)
      - allow easy transcoding in international formats.

   Transcoder are the objects that allow importing of text from streams
   (including "StringStreams") and encoding of falcon strings to output devices.

   By picking the right transcoder, the embedder is able to read
   a text file which is recorded in any encoding and to import it in a
   Falcon String, or to write a falcon string to a file that can be
   then opened in other applications.

   The stream the transcoder receives is not owned by the transcoder:
   the caller must close and dispose of the given stream separatedly.

   As all transcoding operations are character oriented, unless the
   stream is considerabily small, using a BufferedStream is
   highly recommended (unless, of course, using a memory based
   stream like object).

   A convenience function TranscoderFactory is provided to build
   a transcoder for a given character encoding.

   This class has also a read ahead and write-back buffer facilities.
   This allows to build very complex lexers and parsers by using this
   class as an interface to a stream that may not have seek capabilities.
   The buffer is constructed only if needed, so there is no extra
   cost for this facility.

   \note default behavior when transcoding a character that cannot
         be represented in the target encoding is to write a "?"
         (unicode question mark) character.

   \see TranscoderFactory
*/
class FALCON_DYN_CLASS Transcoder: public Stream
{
protected:
   Stream *m_stream;
   bool m_parseStatus;
   bool m_streamOwner;

   Transcoder( Stream *s, bool bOwn );
   Transcoder( const Transcoder &other );

public:

   virtual ~Transcoder();

   virtual t_status status() const { return m_stream->status(); }

   /** Returns the character encoding which is managed by this transcoder.
      Subclasses must reimplement this to return the name of the supported encoding.
   */
   virtual const String encoding() const = 0;


   /** Returns the underlying stream used by this transcoder.
   \return the underlying stream.
   */
   Stream *underlying() const { return m_stream; }

   /** Return encoder status.
      A false status indicates that the last encoding or decoding operation
      wasn't succesful. In example, it may indicate an incorrect UTF-8 sequence
      in utf-8 reading, or an unencodable character in ISO8859 writing.

      Stream I/O error do not set to false this state, and operations may
      return true even if this state get false. In example, ISO8859 encoding
      writes a "?" in case of unencodable character and correctly completes
      the operation, but to inform the user about this fact it sets the
      status to false.

      Iterative get operations (i.e. get string) will be interrupted if status
      is false, and will return false, as the quality of the stream cannot
      be granted.

      Iterative write operations will return true and complete iterations,
      but will set state to false on exit.

      \return false if last get or write operation caused an encoding breaking.
   */
   bool encoderStatus() const { return m_parseStatus; }

   /** Sets the underlying stream.
      To be able to set a stream after that the transcoder has been created.
      This function can set property on the stream. If the owner parameter is
      set to true, the stream pointer will be destroyed when the transcoder
      will be destroyed.

      \note If the previously used stream was owned by this instance,
      it is destroyed here.

      \param s stream to be used as underlying stream
      \param owner true if the stream must be destroyed at transcoder termination.
   */
   void setUnderlying( Stream *s, bool owner=false );

   virtual bool writeString( const String &source, uint32 begin=0, uint32 end = csh::npos );
   virtual bool readString( String &source, uint32 size );

protected:
   virtual int64 seek( int64 pos, e_whence w );

public:

   virtual bool close();
   virtual int64 tell();
   virtual bool truncate( int64 pos=-1 );

   virtual int32 read( void *buffer, int32 size ) { return m_stream->read( buffer, size ); }
   virtual int32 write( const void *buffer, int32 size ) { return m_stream->write( buffer, size ); }
   virtual bool errorDescription( ::Falcon::String &description ) {
      return m_stream->errorDescription( description );
   }

   virtual int32 readAvailable( int32 msecs_timeout, const Sys::SystemData *sysData = 0 ) {
      return m_stream->readAvailable( msecs_timeout, sysData );
   }

   virtual int32 writeAvailable( int32 msecs_timeout, const Sys::SystemData *sysData ) {
      return m_stream->writeAvailable( msecs_timeout, sysData );
   }

   virtual int64 lastError() const
   {
      return m_stream->lastError();
   }

   virtual bool flush();
};

/** EOL Transcoder.
      Under some OSs, and in some stream environment (i.e. TELNET streams,
      HTTP headers etc.), the line terminator is a sequence of characters
      CR+LF. Falcon strings internal encoding sees '\n' (LF) as a line
      terminator marker. As the transcoders are objects meant to translate
      the Falcon internal encoding into a text resource under the rules of
      external encoding, they also provide the facility to translate
      LF character into a CRLF sequence, and to recognize the CRLF sequence
      in read to translate it back into a '\n' line terminator.

      This trancoder can be cascaded with other transcoder; the stream
      used by this transcoder may be another transcoder storing output
      to files.
*/
class FALCON_DYN_CLASS TranscoderEOL: public Transcoder
{
public:
   TranscoderEOL( Stream *s, bool bOwn=false ):
      Transcoder( s, bOwn )
   {}
   TranscoderEOL( const TranscoderEOL &other );

   virtual bool get( uint32 &chr );
   virtual bool put( uint32 chr );
   virtual const String encoding() const { return "EOL"; }
   virtual UserData *clone();
};

/** Transparent byte oriented encoder.
   This encoder writes anything below 256 directly; characters above 256 are
   translated into a '?', but the default can be overridden with the
   substituteChar method.

   This is a good default transcoder to be used in case it is not possibile to
   determine a transcoder.

   Code name for this transcoder is "byte", incase you want to summon it
   throgh TranscoderFactory.
*/
class FALCON_DYN_CLASS TranscoderByte: public Transcoder
{
   byte m_substitute;

public:
   TranscoderByte( Stream *s, bool bOwn=false ):
      Transcoder( s, bOwn ),
      m_substitute( (byte) '?' )
   {}

   TranscoderByte( const TranscoderByte &other );

   /** Set the substitute character.
      This is the character that this transcoder writes insead of
      chars above 255.
      By default it is '?'.
   */
   void substituteChar( char chr ) { m_substitute = (byte) chr; }
   char substituteChar() const { return m_substitute; }

   virtual bool get( uint32 &chr );
   virtual bool put( uint32 chr );
   virtual const String encoding() const { return "byte"; }
   virtual UserData *clone();
};


/** UTF-8 encoding transcoder. */
class FALCON_DYN_CLASS TranscoderUTF8: public Transcoder
{
public:
   TranscoderUTF8( Stream *s, bool bOwn=false ):
      Transcoder( s, bOwn )
   {}

   TranscoderUTF8( const TranscoderUTF8 &other );

   virtual bool get( uint32 &chr );
   virtual bool put( uint32 chr );
   virtual const String encoding() const { return "utf-8"; }
   virtual UserData *clone();
};

/** UTF-16 encoding transcoder. */
class FALCON_DYN_CLASS TranscoderUTF16: public Transcoder
{
public:
   /** Endianity specification, see the constructor. */
   typedef enum {
      e_detect,
      e_le,
      e_be
   } t_endianity;

private:
   t_endianity m_defEndian;
   t_endianity m_streamEndian;
   t_endianity m_hostEndian;

   bool m_bFirstIn;
   bool m_bFirstOut;

protected:
   bool m_bom;

public:

   /** Constructor
      UTF-16 requires specification of endianity with a prefix character.

      On output transcoding, this character is written right before the
      first write() is performed, while on input it is read right before
      the first get() is performed.

      This constructor allows to select an endianity used in I/O. The
      default is e_none; in this case, the transcoder will detect and
      use the system endianity on output.

      In input, the marker read at first get() determines the endianity.
      If the marker can't be found (i.e. if the stream was not at beginning),
      the supplied parameter will be used.

      If the endianity is e_detect, the host system endianity is used.

      The method endianity() can be used to determine the endianity
      of the stream in input that was decided either by reading the marker
      or by the constructor.

      \param s the underlying stream
      \param bOwn own the underlying stream
      \param endianity the endianity for operations.
   */
   TranscoderUTF16( Stream *s, bool bOwn=false, t_endianity endianity = e_detect );
   TranscoderUTF16( const TranscoderUTF16 &other );

   virtual bool get( uint32 &chr );
   virtual bool put( uint32 chr );
   virtual const String encoding() const { return "utf-16"; }
   t_endianity endianity() const { return m_streamEndian; }
   virtual UserData *clone();
};

/** UTF-16LE encoding transcoder. */
class FALCON_DYN_CLASS TranscoderUTF16LE: public TranscoderUTF16
{
public:
   TranscoderUTF16LE( Stream *s, bool bOwn=false ):
      TranscoderUTF16( s, bOwn, e_le )
   {
      m_bom = false;
   }

   virtual const String encoding() const { return "utf-16LE"; }
};

/** UTF-16BE encoding transcoder. */
class FALCON_DYN_CLASS TranscoderUTF16BE: public TranscoderUTF16
{
public:
   TranscoderUTF16BE( Stream *s, bool bOwn=false ):
      TranscoderUTF16( s, bOwn, e_be )
   {
      m_bom = false;
   }

   virtual const String encoding() const { return "utf-16BE"; }
};


/** Base class for codepage like transcoders. */
class FALCON_DYN_CLASS TranscoderISO_CP: public Transcoder
{
protected:
   TranscoderISO_CP( Stream *s, bool bOwn=false ):
      Transcoder( s, bOwn )
   {}

   TranscoderISO_CP( const TranscoderISO_CP &other );

   uint16 *m_directTable;
   uint32 m_dirTabSize;
   CP_ISO_UINT_TABLE *m_reverseTable;
   uint32 m_revTabSize;
public:

   virtual bool get( uint32 &chr );
   virtual bool put( uint32 chr );
};

class FALCON_DYN_CLASS TranscoderCP1252:public TranscoderISO_CP
{
public:
   TranscoderCP1252( const TranscoderCP1252 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderCP1252( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "cp1252"; }
   virtual UserData *clone();
};

/** Latin-1 (ISO8859_1) transcoder. */
class FALCON_DYN_CLASS TranscoderISO8859_1: public TranscoderISO_CP
{
public:
   TranscoderISO8859_1( const TranscoderISO8859_1 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_1( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-1"; }
   virtual UserData *clone();
};

class FALCON_DYN_CLASS TranscoderISO8859_2: public TranscoderISO_CP
{
public:
   TranscoderISO8859_2( const TranscoderISO8859_2 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_2( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-2"; }
   virtual UserData *clone();
};

class FALCON_DYN_CLASS TranscoderISO8859_3: public TranscoderISO_CP
{
public:
   TranscoderISO8859_3( const TranscoderISO8859_3 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_3( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-3"; }
   virtual UserData *clone();
};


class FALCON_DYN_CLASS TranscoderISO8859_4: public TranscoderISO_CP
{
public:
   TranscoderISO8859_4( const TranscoderISO8859_4 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_4( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-4"; }
   virtual UserData *clone();
};

class FALCON_DYN_CLASS TranscoderISO8859_5: public TranscoderISO_CP
{
public:
   TranscoderISO8859_5( const TranscoderISO8859_5 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_5( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-5"; }
   virtual UserData *clone();
};

class FALCON_DYN_CLASS TranscoderISO8859_6: public TranscoderISO_CP
{
public:
   TranscoderISO8859_6( const TranscoderISO8859_6 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_6( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-6"; }
   virtual UserData *clone();
};

class FALCON_DYN_CLASS TranscoderISO8859_7: public TranscoderISO_CP
{
public:
   TranscoderISO8859_7( const TranscoderISO8859_7 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_7( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-7"; }
   virtual UserData *clone();
};

class FALCON_DYN_CLASS TranscoderISO8859_8: public TranscoderISO_CP
{
public:
   TranscoderISO8859_8( const TranscoderISO8859_8 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_8( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-8"; }
   virtual UserData *clone();
};

class FALCON_DYN_CLASS TranscoderISO8859_9: public TranscoderISO_CP
{
public:
   TranscoderISO8859_9( const TranscoderISO8859_9 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_9( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-9"; }
   virtual UserData *clone();
};

class FALCON_DYN_CLASS TranscoderISO8859_10: public TranscoderISO_CP
{
public:
   TranscoderISO8859_10( const TranscoderISO8859_10 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_10( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-10"; }
   virtual UserData *clone();
};

class FALCON_DYN_CLASS TranscoderISO8859_11: public TranscoderISO_CP
{
public:
   TranscoderISO8859_11( const TranscoderISO8859_11 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_11( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-11"; }
   virtual UserData *clone();
};

class FALCON_DYN_CLASS TranscoderISO8859_13: public TranscoderISO_CP
{
public:
   TranscoderISO8859_13( const TranscoderISO8859_13 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_13( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-13"; }
   virtual UserData *clone();
};

class FALCON_DYN_CLASS TranscoderISO8859_14: public TranscoderISO_CP
{
public:
   TranscoderISO8859_14( const TranscoderISO8859_14 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_14( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-14"; }
   virtual UserData *clone();
};

class FALCON_DYN_CLASS TranscoderISO8859_15: public TranscoderISO_CP
{
public:
   TranscoderISO8859_15( const TranscoderISO8859_15 &other ):
      TranscoderISO_CP( other )
   {}

   TranscoderISO8859_15( Stream *s, bool bOwn=false );
   virtual const String encoding() const { return "iso8859-15"; }
   virtual UserData *clone();
};


/** Creates a transcoder for the given encoding.
   If the encoding is not recognized or not supported, null is returned.

   The function can be called without having the stream ready to be
   sure that a certain encoding is supported before opening the stream.
   After that, the returned transcoder must be given a stream with
   the setStream method as soon as possible.

   \param encoding the ISO encoding name for which to build a transcoder
   \param stream optional stream that shall be used by the created transcoder
   \param own set to true if the given stream should be owned (and destroyed) by the transcoder.
*/
FALCON_DYN_SYM Transcoder *TranscoderFactory( const String &encoding, Stream *stream=0, bool own = false );

/** Transcode a string into another.
   This is a convenience function that transcodes a source string into a target string.
   The target string is given a byte manipulator; this means that both the embedders
   and the scripts will be able to access every byte in the resulting string.

   The target string can be written as-is on a stream and the resulting output will
   respect the format specified by the selected encoding.
   \param source the Falcon string to be encoded
   \param encoding the name of the target encoding
   \param target the target where the string will be encoded
   \return false if the target encoding is not supported.
*/

FALCON_DYN_SYM bool TranscodeString( const String &source, const String &encoding, String &target );


/** Transcode an external text string into a Falcon string.
   This is a convenience function that transcodes a source string into a target string.
   The target string is a Falcon string that will accept data a foreign, text encoded
   string transcoding.

   The target string must be re-encoded into something else before being written
   into a text oriented file.

   \param source a text encoded source string
   \param encoding the encoding that is used in the source string
   \param target the target string.
   \return false if the target encoding is not supported.
*/
FALCON_DYN_SYM bool TranscodeFromString( const String &source, const String &encoding, String &target );

/** Determines the default encoding used on the system.
   On success, the parameter is filled with the name of an encoding that can be
   instantiated through TranscodeFactory(). If the function is not able to determine
   the default system encoding, false is returned.
   The transparent "byte" encoding is never returned. It is supposed that, on failure
   the caller should decide wether to use the "byte" encoding or take a sensible action.

   However, "byte" encoding is returned if system encoding is detected to be "C" or "POSIX".

   \param encoding on success will be filled with a FALCON encoding name
   \return true on success, false if the encoding used on the system cannot be determined.
*/
FALCON_DYN_SYM bool GetSystemEncoding( String &encoding );

}

#endif

/* end of transcoding.h */
