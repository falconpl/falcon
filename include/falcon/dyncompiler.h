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
#include <falcon/string.h>

namespace Falcon {

class VMContext;
class String;
class Stream;
class TextReader;
class SynTree;
class Transcoder;
class Item;

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
   DynCompiler(VMContext* ctx);
   ~DynCompiler();

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

   /** Utility to return a single value from a compilation.
    * \param str The code to be compiled
    * \param target Where to place the generated valye
    * \return true if the compiled code is a single-value generating code.
    *
    * This is particularly useful in case you want to compile a code
    * which generates a single value, as, for example
    * \code
    * '"Hello"'
    * '{() > "some code snippet"}'
    * 'fuction test(a,b); > a + b; end'
    * \endcode
    *
    * All the above expression would normally generate a syntactic tree
    * containing a single Expression-Value, which, if evaluated, would
    * return the desired expression.
    *
    * IN other words, compiling something as "fuction test(a,b); > a + b; end"
    * would not return a function, but a syntactic tree that, if evaluated,
    * would then yield the function.
    *
    * This method checks if the generated syntree contains a single expression-value,
    * and if it does, it then returns true and fills the \b target item.
    */
   bool compileValue( Item& target, const String& str );


   /** Utility to return a single value from a compilation.
    * \param stream The stream from which to read code to be compiled
    * \param tr A transcoder to be used to parse the text from the stream
    * \param target Where to place the generated valye
    * \return true if the compiled code is a single-value generating code.
    *
    * \see  bool compileValue( const String& str, Item& target );
    */
   bool compileValue( Item& target, Stream* stream, Transcoder* tr = 0 );

   /** Utility to return a single value from a compilation.
    * \param reader The stream from which to read code to be compiled
    * \param target Where to place the generated valye
    * \return true if the compiled code is a single-value generating code.
    *
    * \see  bool compileValue( const String& str, Item& target );
    */
   bool compileValue( Item& target, TextReader* reader );


   /** Returns the line from which the compilation starts.
    *  \return the line from which the compilation starts.
    */
   int startLine() const { return m_line; }

   /** Changed the initial compilation line
    * \param l New initial line (defaults to 1).
    */
   void startLine( int l ) { m_line = l; }

   /** Returns the symbolic module name used in lexing.
    */
   const String& sourceName() const { return m_name; }
   /** Set the name for lexing compilation.
    *
    * This name is the name of the source as reported in syntactic errors.
    * It defaults to "<internal>"
    */
   void sourceName( const String& value ) { m_name = value; }

private:
   VMContext* m_ctx;
   int m_line;
   String m_name;
};

}

#endif

/* end of dyncompiler.cpp */
