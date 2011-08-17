/*
   FALCON - The Falcon Programming Language.
   FILE: intmode.h

   Interactive mode - step by step dynamic compiler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Apr 2011 21:57:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_INTMODE_H
#define _FALCON_INTMODE_H

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

class VMachine;
class SourceParser;
class SourceLexer;

class TextReader;
class TextWriter;
class StringStream;

class IntMode
{
   IntMode();
   ~IntMode();

   void run( const String& snippet );

private:
   VMachine* m_vm;
   SourceParser* m_parser;
   SourceLexer* m_lexer;

   TextReader* m_reader;
   TextWriter* m_writer;

   //TODO: A PipeMemStream.
   StringStream* m_stream;

   // This class is also used as the "private" pointer.
   class Context;
   Context* m_context;
};

}

#endif /* _FALCON_INTMODE_H */

/* end of intmode.h */
