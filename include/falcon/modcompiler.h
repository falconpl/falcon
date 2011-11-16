/*
   FALCON - The Falcon Programming Language.
   FILE: modcompiler.h

   Module compiler from non-interactive text file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 07 May 2011 16:04:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_MODCOMPILER_H
#define	FALCON_MODCOMPILER_H

#include <falcon/setup.h>
#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/syntree.h>

namespace Falcon {

class Module;
class VMachine;
class StringStream;
class TextReader;

/** Class encapsulating a static module compiler.

 A module compiler is a compiler generating a stand-alone module out of
 a source file.

 A Falcon module can then be serialized to a stream for later usage, or
 can be linked immediately in a virtual machine. Once linked, the module
 cannot be used elsewhere. However, a module can be cloned before linking
 to be used multiple times in a single program.

 A module compiler can be used multiple times.
 */
class FALCON_DYN_CLASS ModCompiler
{

public:
   ModCompiler();
   virtual ~ModCompiler();

   /** Compile the module read from the given TextReader.
    \param input The TextReader where the input is stored.
    \param uri The physical location of the module
    \param name The logical name of the module.
    */
   Module* compile( TextReader* input, const String& uri, const String& name );

   /** Enumerate received errors.
    In case parse returned false, calling this method will provide detailed
    error description for all the errors that have been found.
      \see Enumerator
    */
   inline void enumerateErrors( SourceParser::ErrorEnumerator& e ) const
   {
      m_sp.enumerateErrors( e );
   }
   
   /** Generate an error in case of compilation problems.
    \return an error instance including all the compilation errors or 0 if
    there wasn't any error.
    
    \TODO Return CompileError.
    */
   inline GenericError* makeError() const 
   {
      return m_sp.makeError();
   }

private:

   /** Class used to notify the compiler about relevant facts in parsing. */
   class Context: public ParserContext {
   public:
      Context( ModCompiler* owner );
      virtual ~Context();

      virtual void onInputOver();
      virtual void onNewFunc( Function* function, Symbol* gs=0 );
      virtual void onNewClass( Class* cls, bool bIsObj, Symbol* gs=0 );
      virtual void onNewStatement( Statement* stmt );
      virtual void onLoad( const String& path, bool isFsPath );
      virtual void onImportFrom( const String& path, bool isFsPath, const String& symName,
            const String& nsName, bool bIsNS );
      virtual void onImport(const String& symName );
      virtual void onExport(const String& symName);
      virtual void onDirective(const String& name, const String& value);
      virtual void onGlobal( const String& name );
      virtual Symbol* onUndefinedSymbol( const String& name );
      virtual Symbol* onGlobalDefined( const String& name, bool& bUnique );
      virtual bool onUnknownSymbol( const String& name );
      virtual Expression* onStaticData( Class* cls, void* data );
      virtual void onInheritance( Inheritance* inh  );
      virtual void onRequirement( Requirement* rec );

   private:
      ModCompiler* m_owner;
   };

   SourceParser m_sp;

   /** Used to keep non-transient data. */
   Module* m_module;

   // better for the context to be a pointer, so we can control it's init order.
   Context* m_ctx;
   friend class Context;

   // count of lambda functions
   int m_nLambdaCount;

   // count of anonymous classes
   int m_nClsCount;   
};

}

#endif	/* FALCON_MODCOMPILER_H */

/* end of modcompiler.h */
