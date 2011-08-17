/*
   FALCON - The Falcon Programming Language.
   FILE: generator.h

   Code genartor base class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab giu 5 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_GENERATOR_H
#define FALCON_GENERATOR_H

#include <falcon/setup.h>
#include <falcon/basealloc.h>

namespace Falcon
{

class Stream;
class Module;
class SourceTree;

class FALCON_DYN_CLASS Generator: public BaseAlloc
{
protected:
  Stream *m_out;

public:
   /** Creates the generator.
      Although a stream is provided as a parameter, ownership is not taken.
      The stream is still open and available after generator destruction.
      \param out the stream where output of this generator will be sent.
   */
   Generator( Stream *out ):
      m_out( out )
   {}

   virtual ~Generator()
   {}

   virtual void generate( const SourceTree *st ) = 0;
};

}
#endif

/* end of generator.h */
