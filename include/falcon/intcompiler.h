/*
   FALCON - The Falcon Programming Language.
   FILE: intcompiler.h

   Interactive compiler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 07 May 2011 16:04:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_INTCOMPILER_H
#define FALCON_INTCOMPILER_H

#include <falcon/setup.h>
#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/syntree.h>

namespace Falcon {

class Module;
class VMachine;
class StringStream;
class TextReader;
class TextWriter;

/** Class encapsulating an interactive compiler.

A Falcon interactive compiler is a compiler that reads text and interprets
it as a Falcon program as it's being completed. It's meant to support the
interactive mode, where programs are tested or modules are used from the
command prompt.

The interactive compiler actually lives inside a host context in a live
process of a virtual machine. A PStep constantly invokes the interactive
compiler and eventually manages error that are raised from within.

 */
class FALCON_DYN_CLASS IntCompiler
{

public:
   IntCompiler( bool allowDirectives = true );
   virtual ~IntCompiler();

   typedef enum {
      /** The statement is incomplete; more input is needed. */
      e_incomplete,
      /** The statement is an (auto) expression that should be evaluated (and result told to the user). */
      e_expression,
      /** The statement is a single call expression; typically, if it returns nil the user should not be told. */
      e_expr_call,
      /** The statement is not an expression and it doesn't require to generate a result */
      e_statement,
      /** The statement was a mantra definition (code will be 0). */
      e_definition
   } t_compile_status;

   /** Compile available data.
    This method returns as soon as a complete statement, definition or expression is parsed,
    or as soon as the input stream is exhausted (or, if it's non-blocking, as soon as
    the stream has nothing available).

    The method might return a valid data even in case some errors were detected.
    */
   t_compile_status compileNext( TextReader* input, SynTree*& code, Mantra*& definition );
   
   /** Clear the currently compiled items.

    In case of incomplete parsing, calling this method clears the tree read up
    to date.
    */
   void resetTree();

   /** Tell whether the current code is self-consistent or needs more data.
    \return true if the curent compiler is in a clean state, false if it's
    waiting for more data before processing the input.
    */
   bool isComplete() const;

   bool areDirectivesAllowed() const { return m_bAllowDirective; }
   void setDirectivesAllowed(bool mode ) { m_bAllowDirective = mode; }

   /** helper to generate Interactive Compiler errors. */
   void throwCompileErrors() const;

   /** True if the last compileNext() generated some errors. */
   bool hasErrors() const;

   void setCompilationContext( Function * function, Module* mod, VMContext* ctx );
   Function* getFunction() const { return m_compf; }
   Module* getModule() const { return m_mod; }

private:

   /** Class used to notify the compiler about relevant facts in parsing. */
   class Context: public ParserContext {
   public:
      Context( IntCompiler* owner );
      virtual ~Context();

      virtual void onInputOver();
      virtual void onNewFunc( Function* function );
      virtual void onNewClass( Class* cls, bool bIsObj );
      virtual void onNewStatement( Statement* stmt );
      virtual void onLoad( const String& path, bool isFsPath );
      virtual bool onImportFrom( ImportDef* def );
      virtual void onExport(const String& symName);
      virtual void onDirective(const String& name, const String& value);
      virtual void onGlobal( const String& name );
      virtual Variable* onGlobalDefined( const String& name, bool& bUnique );
      virtual Variable* onGlobalAccessed( const String& name );
      virtual Item* getVariableValue( Variable* var );
      virtual void onRequirement( Requirement* rec );

   private:
      IntCompiler* m_owner;
   };
   
   // adds a compiler error for later throwing.
   void addError( Error* e );


   SourceParser m_sp;
   SynTree* m_currentTree;
   Mantra* m_currentMantra;

   // Should we allow directives or not.
   bool m_bAllowDirective;

   Function* m_compf;
   Module* m_mod;
   VMContext* m_vmctx;

   // better for the context to be a pointer, so we can control it's init order.
   Context* m_ctx;
   Parsing::Lexer* m_lexer;
   friend class Context;
};

}

#endif	/* FALCON_INTCOMPILER_H */

/* end of intcompiler.h */
