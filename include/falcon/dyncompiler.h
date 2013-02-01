/*
   FALCON - The Falcon Programming Language.
   FILE: dyncompiler.cpp

   Falcon core module -- Compile function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 11:12:52 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DYNCOMPILER_H_
#define _FALCON_DYNCOMPILER_H_


#include <falcon/setup.h>
#include <falcon/dyncompiler.h>

namespace Falcon {

class VMContext;
class String;
class Stream;
class TextReader;
class SynTree;
class Transcoder;

/** Dynamic compiler.
 * This compiler is meant to compile some dynamic code from a string
 * or from a stream.
 *
 * It creates (or uses) a SynTree which contains the dynamically
 * compiled code.
 *
 * All the coded created by this compiler is dynamic and
 * never references any module, global variable or external
 * variable explicitly.
 *
 * Directives with module-wide meaning (including attributes and
 * internationalized i"" strings) are signaled as errors.
 */

class FALCON_DYN_CLASS DynCompiler
{
public:
   DynCompiler(VMContext* ctx):
      m_ctx(ctx)
   {}

   ~DynCompiler() {}

   /** Compiles a dynamic code coming from a falcon string.
    * \param reader The source input.
    * \param Target the syntree where to place the statements found in the
    *        compilation.
    * \return target if the parameter is not zero, or a new syntree.
    * Notice that if the transcoder is 0, a C transcoder is used.
    */
   SynTree* compile( const String& str, SynTree* target = 0 );

   /** Compiles a dynamic code coming from a stream.
    * \param reader The source input.
    * \param Target the syntree where to place the statements found in the
    *        compilation.
    * \param Transcoder the transcoder used to read the stream.
    * \return target if the parameter is not zero, or a new syntree.
    * Notice that if the transcoder is 0, a C transcoder is used.
    */
   SynTree* compile( Stream* stream, Transcoder* tr = 0, SynTree* target = 0 );

   /** Compiles a dynamic code coming from a text reader.
    * \param reader The source input.
    * \param Target the syntree where to place the statements found in the
    *        compilation.
    * \return target if the parameter is not zero, or a new syntree.
    */
   SynTree* compile( TextReader* reader, SynTree* target = 0 );

private:
   VMContext* m_ctx;
};

}

#endif

/* end of dyncompiler.cpp */
