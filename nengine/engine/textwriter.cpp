/*
   FALCON - The Falcon Programming Language.
   FILE: textreader.cpp

   Text-oriented stream reader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Mar 2011 11:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/textwriter.h>
#include <falcon/stream.h>
#include <falcon/engine.h>
#include <falcon/transcoder.h>
#include <falcon/encodingerror.h>

#include <string.h>

#include <list>


namespace Falcon {

TextWriter::TextWriter( Stream* stream, bool bOwn ):
   Writer( stream, bOwn ),
{
   m_encoder = Engine::instance()->getTranscoder("C");
}


TextWriter::TextWriter( Stream* stream, Transcoder* decoder, bool bOwn ):
   Writer( stream, bOwn ),
   m_encoder( decoder )
{
}


TextWriter::~TextWriter()
{
}

void TextWriter::setEncoding( Transcoder* decoder )
{
   m_encoder = decoder;
}

bool TextWriter::write( String& str, length_t start, length_t count )
{
}

bool TextWriter::writeLine( String& str, bool bFlush, length_t start, length_t count )
{
}


void TextWriter::putChar( char_t chr )
{
}


}

/* end of textreader.cpp */
