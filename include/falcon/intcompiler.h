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

 The interactive compiler takes a Virtual Machine that is then use to execute
 the code that is getting completed in the meanwhile. It never creates a module
 out of the code it's compiling, and usually discards it after it's executed.

 Functions and classes declared in the process are stored directly in the
 compiler itself.
 */
class FALCON_DYN_CLASS IntCompiler
{

public:
   IntCompiler( VMachine *vm );
   virtual ~IntCompiler();

   typedef enum compile_status_t {
      ok_t,
      incomplete_t,
      eval_t,
      eval_direct_t
   } compile_status;

   compile_status compileNext( const String& value );
   
   /** Clear the currently compiled items.

    In case of incomplete parsing, calling this method clears the tree read up
    to date.
    */
   void resetTree();

   /** Tell wether the current code is self-consistent or needs more data.
    \return true if the curent compiler is in a clean state, false if it's
    waiting for more data before processing the input.
    */
   bool isComplete() const;
   
   /** Completely resets the current context of the VM of the interactive compiler.
    To be used after a error to put the current VM context in initial
    state.
    
    */
   void resetVM();
   
private:

   /** Class used to notify the intcompiler about relevant facts in parsing. */
   class Context: public ParserContext {
   public:
      Context( IntCompiler* owner );
      virtual ~Context();
      
      virtual void onInputOver();
      virtual void onNewFunc( Function* function, Symbol* gs=0 );
      virtual void onNewClass( Class* cls, bool bIsObj, Symbol* gs=0 );
      virtual void onNewStatement( Statement* stmt );
      virtual void onLoad( const String& path, bool isFsPath );
      virtual bool onImportFrom( ImportDef* def );
      virtual void onExport(const String& symName);
      virtual void onDirective(const String& name, const String& value);
      virtual void onGlobal( const String& name );
      virtual Symbol* onUndefinedSymbol( const String& name );
      virtual Symbol* onGlobalDefined( const String& name, bool& bUnique );
      virtual Expression* onStaticData( Class* cls, void* data );
      virtual void onInheritance( Inheritance* inh  );
      virtual void onRequirement( Requirement* rec );

   private:
      IntCompiler* m_owner;
   };

   // helper to generate Interactive Compiler errors.
   void throwCompileErrors() const;
   
   // adds a compiler error for later throwing.
   void addError( Error* e );

   SourceParser m_sp;
   VMachine* m_vm;
   SynTree* m_currentTree;

   StringStream* m_stream;
   TextWriter* m_writer;

   /** Used to keep non-transient data. */
   Module* m_module;
   Function* m_main;
   
   // better for the context to be a pointer, so we can control it's init order.
   Context* m_ctx;
   friend class Context;

   
};

}

#endif	/* FALCON_INTCOMPILER_H */

/* end of intcompiler.h */